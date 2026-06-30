"""Load mCODE/Synthea FHIR patient bundles into flat dicts for RoleGuard."""

from __future__ import annotations

import json
import warnings
from datetime import datetime
from pathlib import Path

# Snapshot window: associate clinical resources with the most recent encounter.
SNAPSHOT_WINDOW_DAYS = 90
LAB_RESULTS_WINDOW_DAYS = 180

ACTIVE_MEDICATION_STATUSES = {"active", "on-hold", "unknown"}

SKIPPED_LAB_LABELS = {"History and physical note"}

ONCOLOGY_ENCOUNTER_TERMS = (
    "oncology",
    "cancer",
    "chemotherapy",
    "radiation",
    "biopsy",
    "mastectomy",
    "lumpectomy",
    "prostatectomy",
)

CANCER_DIAGNOSIS_TERMS = ("malignant neoplasm", "carcinoma", "cancer")


def _parse_datetime(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None


def _ref_id(reference: str | None) -> str:
    if not reference:
        return ""
    if reference.startswith("urn:uuid:"):
        return reference.rsplit(":", 1)[-1]
    return reference.rsplit("/", 1)[-1]


def _resources_by_type(bundle: dict) -> dict[str, list[dict]]:
    grouped: dict[str, list[dict]] = {}
    for entry in bundle.get("entry", []):
        resource = entry.get("resource", {})
        resource_type = resource.get("resourceType")
        if resource_type:
            grouped.setdefault(resource_type, []).append(resource)
    return grouped


def _build_practitioner_lookup(resources: dict[str, list[dict]]) -> dict[str, str]:
    lookup: dict[str, str] = {}
    for practitioner in resources.get("Practitioner", []):
        practitioner_id = practitioner.get("id", "")
        names = practitioner.get("name", [])
        if names:
            name = names[0]
            given = " ".join(name.get("given", []))
            family = name.get("family", "")
            prefix = " ".join(name.get("prefix", []))
            display = " ".join(part for part in [prefix, given, family] if part).strip()
        else:
            display = practitioner_id
        lookup[practitioner_id] = display or practitioner_id
        lookup[f"urn:uuid:{practitioner_id}"] = display or practitioner_id
    return lookup


def _get_encounter_department(encounter: dict) -> str:
    encounter_types = encounter.get("type", [])
    if encounter_types:
        text = encounter_types[0].get("text")
        if text:
            return text
        coding = encounter_types[0].get("coding", [])
        if coding:
            return coding[0].get("display") or coding[0].get("code", "")
    return encounter.get("class", {}).get("code", "")


def _get_clinician_id(encounter: dict, practitioners: dict[str, str]) -> str:
    for participant in encounter.get("participant", []):
        individual = participant.get("individual", {})
        reference = individual.get("reference", "")
        display = individual.get("display")
        if display:
            return display
        resolved = practitioners.get(reference) or practitioners.get(_ref_id(reference))
        if resolved:
            return resolved

    service_provider = encounter.get("serviceProvider", {})
    if service_provider.get("display"):
        return service_provider["display"]
    reference = service_provider.get("reference", "")
    return practitioners.get(reference) or practitioners.get(_ref_id(reference)) or reference


def _condition_text(condition: dict) -> str:
    return condition.get("code", {}).get("text", "").lower()


def _is_breast_cancer_condition(condition: dict) -> bool:
    text = _condition_text(condition)
    return "breast" in text


def _is_cancer_condition(condition: dict) -> bool:
    text = _condition_text(condition)
    if _is_breast_cancer_condition(condition):
        return True
    return any(term in text for term in CANCER_DIAGNOSIS_TERMS)


def _extract_cancer_diagnosis(
    conditions: list[dict],
) -> tuple[str, str, bool, list[dict]]:
    """Return diagnosis, code, is_breast flag, and matched cancer conditions."""
    cancer_conditions = [c for c in conditions if _is_cancer_condition(c)]
    if not cancer_conditions:
        return "", "", False, []

    breast_conditions = [c for c in cancer_conditions if _is_breast_cancer_condition(c)]
    if breast_conditions:
        selected_pool = breast_conditions
        is_breast = True
    else:
        selected_pool = cancer_conditions
        is_breast = False

    selected_pool.sort(
        key=lambda c: c.get("recordedDate") or c.get("onsetDateTime") or "",
        reverse=True,
    )
    condition = selected_pool[0]
    code = condition.get("code", {})
    diagnosis = code.get("text", "")
    coding = code.get("coding", [])
    # NOTE: diagnosis_code is SNOMED CT in this dataset, not ICD-10.
    diagnosis_code = coding[0].get("code", "") if coding else ""
    return diagnosis, diagnosis_code, is_breast, cancer_conditions


def _cancer_condition_refs(cancer_conditions: list[dict]) -> tuple[set[str], set[str]]:
    condition_ids: set[str] = set()
    condition_codes: set[str] = set()
    for condition in cancer_conditions:
        condition_id = condition.get("id", "")
        if condition_id:
            condition_ids.add(condition_id)
        for coding in condition.get("code", {}).get("coding", []):
            code = coding.get("code")
            if code:
                condition_codes.add(code)
    return condition_ids, condition_codes


def _encounter_type_text(encounter: dict) -> str:
    parts: list[str] = []
    for encounter_type in encounter.get("type", []):
        if encounter_type.get("text"):
            parts.append(encounter_type["text"])
        for coding in encounter_type.get("coding", []):
            if coding.get("display"):
                parts.append(coding["display"])
    return " ".join(parts).lower()


def _codeable_concepts_text(codeables: list[dict]) -> str:
    parts: list[str] = []
    for concept in codeables:
        if concept.get("text"):
            parts.append(concept["text"])
        for coding in concept.get("coding", []):
            if coding.get("display"):
                parts.append(coding["display"])
            if coding.get("code"):
                parts.append(coding["code"])
    return " ".join(parts).lower()


def _encounter_relates_to_cancer(
    encounter: dict,
    cancer_conditions: list[dict],
) -> bool:
    condition_ids, condition_codes = _cancer_condition_refs(cancer_conditions)

    for reason_ref in encounter.get("reasonReference", []):
        ref_id = _ref_id(reason_ref.get("reference"))
        if ref_id in condition_ids:
            return True

    for reason_code in encounter.get("reasonCode", []):
        reason_text = _codeable_concepts_text([reason_code])
        for coding in reason_code.get("coding", []):
            if coding.get("code") in condition_codes:
                return True
        if any(term in reason_text for term in CANCER_DIAGNOSIS_TERMS):
            return True
        if "breast" in reason_text:
            return True

    type_text = _encounter_type_text(encounter)
    return any(term in type_text for term in ONCOLOGY_ENCOUNTER_TERMS)


def _select_snapshot_encounter(
    encounters: list[dict],
    cancer_conditions: list[dict],
    patient_id: str,
) -> dict:
    if not encounters:
        raise ValueError("No Encounter resources found")

    sorted_encounters = sorted(
        encounters,
        key=lambda enc: enc.get("period", {}).get("start", ""),
        reverse=True,
    )

    oncology_encounters = [
        enc
        for enc in sorted_encounters
        if _encounter_relates_to_cancer(enc, cancer_conditions)
    ]

    if oncology_encounters:
        return oncology_encounters[0]

    warnings.warn(
        f"Patient {patient_id}: no oncology-specific encounter found; "
        "using most recent encounter overall — snapshot may not be oncology-specific",
        stacklevel=2,
    )
    return sorted_encounters[0]


def _resource_date(resource: dict, *fields: str) -> datetime | None:
    for field in fields:
        if field in resource:
            parsed = _parse_datetime(resource[field])
            if parsed:
                return parsed
    for period_field in ("performedPeriod", "effectivePeriod", "period"):
        period = resource.get(period_field, {})
        if isinstance(period, dict) and period.get("start"):
            parsed = _parse_datetime(period["start"])
            if parsed:
                return parsed
    return None


def _matches_encounter_snapshot(
    resource: dict,
    encounter_id: str,
    encounter_start: datetime | None,
    window_days: int = SNAPSHOT_WINDOW_DAYS,
) -> bool:
    encounter_ref = resource.get("encounter", {}).get("reference", "")
    if encounter_id and encounter_id in encounter_ref:
        return True

    if encounter_start is None:
        return False

    resource_dt = _resource_date(
        resource,
        "authoredOn",
        "performedDateTime",
        "effectiveDateTime",
        "issued",
        "started",
        "recordedDate",
    )
    if resource_dt is None:
        return False

    encounter_naive = encounter_start.replace(tzinfo=None)
    resource_naive = resource_dt.replace(tzinfo=None)
    return abs((resource_naive - encounter_naive).days) <= window_days


def _format_observation_result(resource: dict) -> str:
    code = resource.get("code", {})
    label = code.get("text") or ""
    if not label and code.get("coding"):
        label = code["coding"][0].get("display") or code["coding"][0].get("code", "")

    value = None
    if "valueQuantity" in resource:
        qty = resource["valueQuantity"]
        value = f"{qty.get('value', '')} {qty.get('unit', '')}".strip()
    elif "valueString" in resource:
        value = resource["valueString"]
    elif "valueCodeableConcept" in resource:
        value = resource["valueCodeableConcept"].get("text", "")

    if value:
        return f"{label}: {value}"
    return label


def _format_diagnostic_report(resource: dict) -> str:
    code = resource.get("code", {})
    label = code.get("text") or ""
    if not label and code.get("coding"):
        label = code["coding"][0].get("display") or code["coding"][0].get("code", "")
    return label


def _is_lab_like_observation(observation: dict) -> bool:
    categories = observation.get("category", [])
    for category in categories:
        for coding in category.get("coding", []):
            if coding.get("code") in {"laboratory", "exam", "imaging", "procedure"}:
                return True
    return False


def _extract_imaging_descriptions(imaging_studies: list[dict]) -> list[str]:
    descriptions: list[str] = []
    for study in imaging_studies:
        for series in study.get("series", []):
            parts: list[str] = []
            modality = series.get("modality", {})
            if modality.get("display"):
                parts.append(modality["display"])
            body_site = series.get("bodySite", {})
            if body_site.get("display"):
                parts.append(body_site["display"])
            for instance in series.get("instance", []):
                if instance.get("title"):
                    parts.append(instance["title"])
            if parts:
                descriptions.append(" - ".join(parts))
            elif study.get("description"):
                descriptions.append(study["description"])
    return descriptions


def _parse_patient_bundle(bundle: dict, filepath: str = "") -> tuple[dict, bool]:
    """Parse a FHIR bundle into a flat snapshot dict and breast-cancer flag."""
    resources = _resources_by_type(bundle)
    patients = resources.get("Patient", [])
    if not patients:
        raise ValueError(f"No Patient resource found in {filepath}")

    patient = patients[0]
    patient_id = patient.get("id", "")

    conditions = resources.get("Condition", [])
    diagnosis, diagnosis_code, is_breast, cancer_conditions = _extract_cancer_diagnosis(
        conditions
    )

    encounters = resources.get("Encounter", [])
    if not encounters:
        raise ValueError(f"No Encounter resources found in {filepath}")

    recent_encounter = _select_snapshot_encounter(encounters, cancer_conditions, patient_id)
    encounter_id = recent_encounter.get("id", "")
    encounter_start = _parse_datetime(recent_encounter.get("period", {}).get("start"))

    practitioners = _build_practitioner_lookup(resources)

    medications: list[str] = []
    near_medications: list[dict] = []
    for med in resources.get("MedicationRequest", []):
        if _matches_encounter_snapshot(med, encounter_id, encounter_start):
            near_medications.append(med)

    active_near = [
        med for med in near_medications if med.get("status") in ACTIVE_MEDICATION_STATUSES
    ]
    selected_medications = active_near or near_medications
    if not selected_medications:
        selected_medications = [
            med
            for med in resources.get("MedicationRequest", [])
            if med.get("status") in ACTIVE_MEDICATION_STATUSES
        ]

    for med in selected_medications:
        concept = med.get("medicationCodeableConcept", {})
        text = concept.get("text")
        if text:
            medications.append(text)

    procedure_codes: list[str] = []
    for procedure in resources.get("Procedure", []):
        if not _matches_encounter_snapshot(procedure, encounter_id, encounter_start):
            continue
        code = procedure.get("code", {})
        coding = code.get("coding", [])
        proc_code = coding[0].get("code", "") if coding else ""
        proc_text = code.get("text", "")
        if proc_code and proc_text:
            procedure_codes.append(f"{proc_code}: {proc_text}")
        elif proc_code:
            procedure_codes.append(proc_code)
        elif proc_text:
            procedure_codes.append(proc_text)

    lab_results: list[str] = []
    for observation in resources.get("Observation", []):
        if not _matches_encounter_snapshot(
            observation, encounter_id, encounter_start, LAB_RESULTS_WINDOW_DAYS
        ):
            continue
        if not _is_lab_like_observation(observation):
            continue
        formatted = _format_observation_result(observation)
        if formatted and formatted not in SKIPPED_LAB_LABELS:
            lab_results.append(formatted)

    for report in resources.get("DiagnosticReport", []):
        if not _matches_encounter_snapshot(
            report, encounter_id, encounter_start, LAB_RESULTS_WINDOW_DAYS
        ):
            continue
        formatted = _format_diagnostic_report(report)
        if formatted and formatted not in SKIPPED_LAB_LABELS:
            lab_results.append(formatted)

    imaging_studies = [
        study
        for study in resources.get("ImagingStudy", [])
        if _matches_encounter_snapshot(study, encounter_id, encounter_start)
    ]
    imaging = _extract_imaging_descriptions(imaging_studies)

    return {
        "patient_id": patient_id,
        "diagnosis": diagnosis,
        "diagnosis_code": diagnosis_code,
        "medication": medications,
        "procedure_code": procedure_codes,
        "lab_results": lab_results,
        "imaging": imaging,
        "department": _get_encounter_department(recent_encounter),
        "appointment_time": recent_encounter.get("period", {}).get("start", ""),
        "clinician_id": _get_clinician_id(recent_encounter, practitioners),
    }, is_breast


def load_patient(filepath: str) -> dict:
    """Parse a single FHIR patient bundle into a flat snapshot dict."""
    with open(filepath, encoding="utf-8") as handle:
        bundle = json.load(handle)
    patient, _ = _parse_patient_bundle(bundle, filepath)
    return patient


def load_all_patients(
    directory: str,
    limit: int | None = None,
) -> dict[str, list[dict]]:
    """Load all patient bundles, grouped by breast vs other cancer cohorts."""
    patient_dir = Path(directory)
    files = sorted(patient_dir.glob("*.json"))
    if limit is not None:
        files = files[:limit]

    breast_cancer: list[dict] = []
    other_cancer: list[dict] = []
    failed = 0

    for filepath in files:
        try:
            with open(filepath, encoding="utf-8") as handle:
                bundle = json.load(handle)
            patient, is_breast = _parse_patient_bundle(bundle, str(filepath))
            if is_breast:
                breast_cancer.append(patient)
            else:
                other_cancer.append(patient)
        except Exception as exc:
            failed += 1
            print(f"Failed to load {filepath.name}: {exc}")

    print(
        f"Loaded {len(breast_cancer)} breast cancer, "
        f"{len(other_cancer)} other cancer, {failed} failed"
    )
    return {"breast_cancer": breast_cancer, "other_cancer": other_cancer}


if __name__ == "__main__":
    import pprint

    data_dir = Path(__file__).resolve().parent.parent / "data" / "scenarios" / "all_patients"
    loaded = load_all_patients(str(data_dir))

    if loaded["breast_cancer"]:
        print("\nExample breast cancer patient:")
        pprint.pp(loaded["breast_cancer"][0])

    if loaded["other_cancer"]:
        print("\nExample other cancer patient:")
        pprint.pp(loaded["other_cancer"][0])
