"""Run the cyclic pipeline on 10 breast cancer patients (baseline + protected)."""

from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

warnings.filterwarnings("ignore")

from langchain_ollama import ChatOllama

from src.pipeline import run_pipeline
from src.synthea_loader import load_all_patients

OUTPUT_PATH = _ROOT / "data" / "results" / "pipeline_10patients.json"
NUM_PATIENTS = 10


def count_violations(state: dict, phase: str) -> int:
    """Count RoleLeak entries for a given phase that have output violations."""
    count = 0
    for item in state.get("roleleak_violations") or []:
        if item.get("phase") != phase:
            continue
        measurement = item.get("measurement") or {}
        if measurement.get("has_output_violation"):
            count += 1
        count += len(measurement.get("non_permitted_categories_mentioned") or [])
    return count


def save_results(results: list[dict]) -> None:
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2)
        handle.flush()


def main() -> None:
    llm = ChatOllama(model="llama3")
    patients = load_all_patients(str(_ROOT / "data" / "scenarios" / "all_patients"))[
        "breast_cancer"
    ][:NUM_PATIENTS]

    results: list[dict] = []
    print(f"Running pipeline on {len(patients)} patients...", flush=True)

    for index, patient in enumerate(patients, start=1):
        patient_id = patient.get("patient_id", f"patient_{index}")
        print(f"\n[{index}/{len(patients)}] Patient {patient_id}", flush=True)

        print("  Running baseline...", flush=True)
        baseline_state = run_pipeline(patient, llm, baseline_mode=True)

        print("  Running protected...", flush=True)
        protected_state = run_pipeline(patient, llm, baseline_mode=False)

        result = {
            "patient_id": patient_id,
            "baseline_prefilter": count_violations(baseline_state, "pre_filter"),
            "protected_prefilter": count_violations(protected_state, "pre_filter"),
            "protected_postfilter": count_violations(protected_state, "post_filter"),
            "clarification_rounds": int(protected_state.get("clarification_round") or 0),
            "scheduling_input": str(protected_state.get("scheduling_input") or "")[:200],
            "billing_input": str(protected_state.get("billing_input") or "")[:200],
        }
        results.append(result)
        save_results(results)

        print(
            f"  baseline_prefilter={result['baseline_prefilter']} "
            f"protected_prefilter={result['protected_prefilter']} "
            f"protected_postfilter={result['protected_postfilter']} "
            f"clarification_rounds={result['clarification_rounds']}",
            flush=True,
        )
        print(f"  Saved progress to {OUTPUT_PATH}", flush=True)

    print("Done!", flush=True)


if __name__ == "__main__":
    main()
