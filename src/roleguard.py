"""RoleGuard access control middleware for the multi-agent healthcare pipeline."""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.roleleak import measure_boundary_leakage, _non_empty_categories
from src.roles import CATEGORIES, PERMISSIONS, PIPELINE, get_permitted_only

logger = logging.getLogger(__name__)


def apply_roleguard_with_audit(patient_data: dict, agent_role: str) -> tuple[dict, dict]:
    """
    Filter patient data to permitted categories and return an audit log.
    """
    categories_in_full_record = _non_empty_categories(patient_data)
    categories_permitted = sorted(PERMISSIONS.get(agent_role, set()))
    filtered_data = get_permitted_only(patient_data, agent_role)
    categories_passed_through = _non_empty_categories(filtered_data)
    categories_filtered_out = [
        category
        for category in categories_in_full_record
        if category not in PERMISSIONS.get(agent_role, set())
    ]
    filter_rate = (
        len(categories_filtered_out) / len(categories_in_full_record)
        if categories_in_full_record
        else 0.0
    )

    audit_log = {
        "agent_role": agent_role,
        "categories_in_full_record": categories_in_full_record,
        "categories_permitted": categories_permitted,
        "categories_filtered_out": categories_filtered_out,
        "categories_passed_through": categories_passed_through,
        "filter_rate": filter_rate,
    }
    return filtered_data, audit_log


def apply_roleguard(patient_data: dict, agent_role: str) -> dict:
    """
    Filter patient data to only categories permitted for the agent role.
    """
    filtered_data, audit_log = apply_roleguard_with_audit(patient_data, agent_role)
    logger.info(
        "RoleGuard filter role=%s present=%s permitted=%s filtered_out=%s passed=%s",
        agent_role,
        audit_log["categories_in_full_record"],
        audit_log["categories_permitted"],
        audit_log["categories_filtered_out"],
        audit_log["categories_passed_through"],
    )
    return filtered_data


def run_roleguard_pipeline(
    patient: dict,
    agents: dict,
    llm,
) -> dict:
    """
    Run one patient through the pipeline with RoleGuard active at each boundary.
    """
    results: dict = {"patient_id": patient.get("patient_id", "")}

    for role in PIPELINE:
        filtered_data, audit = apply_roleguard_with_audit(patient, role)
        agent_result = agents[role](filtered_data, llm)
        boundary = measure_boundary_leakage(agent_result["received_data"], role)

        results[role] = {
            "filtered_data": filtered_data,
            "audit": audit,
            "boundary": boundary,
            "output": agent_result["output"],
        }

    return results


def _write_record(output_path: Path, record: dict) -> None:
    with open(output_path, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(record) + "\n")
        handle.flush()


def _init_benchmark_summary() -> dict:
    return {
        role: {
            "boundary_violation_patients": 0,
            "total_filter_rate": 0.0,
            "task_completed_patients": 0,
        }
        for role in PIPELINE
    }


def _compute_benchmark_summary(results: list[dict], patient_count: int) -> dict:
    summary = _init_benchmark_summary()

    for result in results:
        for role in PIPELINE:
            role_result = result[role]
            boundary = role_result["boundary"]
            audit = role_result["audit"]
            output = role_result["output"]

            if boundary["has_violation"]:
                summary[role]["boundary_violation_patients"] += 1
            summary[role]["total_filter_rate"] += audit["filter_rate"]
            if output and str(output).strip():
                summary[role]["task_completed_patients"] += 1

    formatted: dict = {}
    for role in PIPELINE:
        role_summary = summary[role]
        boundary_rate = (
            role_summary["boundary_violation_patients"] / patient_count * 100
            if patient_count
            else 0.0
        )
        avg_filter_rate = (
            role_summary["total_filter_rate"] / patient_count
            if patient_count
            else 0.0
        )
        task_completion_rate = (
            role_summary["task_completed_patients"] / patient_count * 100
            if patient_count
            else 0.0
        )
        formatted[role] = {
            "boundary_violation_rate_pct": boundary_rate,
            "avg_filter_rate": avg_filter_rate,
            "task_completion_rate_pct": task_completion_rate,
        }
    return formatted


def _print_benchmark_summary(summary: dict, patient_count: int) -> None:
    print("\n" + "=" * 60)
    print("RoleGuard Protected Pipeline Summary")
    print("=" * 60)
    print(f"Patients processed: {patient_count}\n")

    for role in PIPELINE:
        role_summary = summary[role]
        print(f"{role.upper()}:")
        print(
            f"  Boundary violation rate: {role_summary['boundary_violation_rate_pct']:.1f}%"
        )
        print(f"  Avg filter rate:         {role_summary['avg_filter_rate']:.2f}")
        print(
            f"  Task completion rate:    {role_summary['task_completion_rate_pct']:.1f}%"
        )
        print()


def run_roleguard_benchmark(
    patients: list[dict],
    agents: dict,
    llm,
    output_file: str = "data/results/roleguard_protected.jsonl",
    limit: int | None = None,
) -> dict:
    """
    Run all patients through the RoleGuard-protected pipeline.
    """
    selected_patients = patients[:limit] if limit is not None else patients
    total = len(selected_patients)

    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists():
        output_path.unlink()

    results: list[dict] = []

    for index, patient in enumerate(selected_patients, start=1):
        patient_id = patient.get("patient_id", f"patient_{index}")
        print(f"Processing patient {index}/{total} ({patient_id})...")

        pipeline_result = run_roleguard_pipeline(patient, agents, llm)
        results.append(pipeline_result)
        _write_record(output_path, pipeline_result)

        for role in PIPELINE:
            boundary = pipeline_result[role]["boundary"]
            print(
                f"  {role}: violations={boundary['violations']}, "
                f"filter_rate={pipeline_result[role]['audit']['filter_rate']:.2f}"
            )

        if index % 10 == 0:
            print(f"Processed {index}/{total} patients...")

    summary = _compute_benchmark_summary(results, total)
    _print_benchmark_summary(summary, total)
    return summary


if __name__ == "__main__":
    import warnings

    from langchain_ollama import ChatOllama

    from agents.billing import billing_agent
    from agents.clinical import clinical_agent
    from agents.scheduling import scheduling_agent
    from src.synthea_loader import load_all_patients

    warnings.filterwarnings("ignore")
    logging.basicConfig(level=logging.INFO)

    llm = ChatOllama(model="llama3")
    data_dir = _PROJECT_ROOT / "data" / "scenarios" / "all_patients"
    patients = load_all_patients(str(data_dir))["breast_cancer"]

    agents = {
        "clinical": clinical_agent,
        "scheduling": scheduling_agent,
        "billing": billing_agent,
    }

    summary = run_roleguard_benchmark(
        patients=patients,
        agents=agents,
        llm=llm,
        output_file=str(_PROJECT_ROOT / "data" / "results" / "roleguard_protected.jsonl"),
        limit=5,
    )

    first_result = json.loads(
        Path(_PROJECT_ROOT / "data" / "results" / "roleguard_protected.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()[0]
    )

    print("=" * 60)
    print("Example: What Each Agent Received (patient 1)")
    print("=" * 60)
    for role in PIPELINE:
        print(f"\n{role.upper()} filtered_data:")
        print(json.dumps(first_result[role]["filtered_data"], indent=2))

    print("\n" + "=" * 60)
    print("Example: Agent Outputs (patient 1, truncated)")
    print("=" * 60)
    for role in PIPELINE:
        output = first_result[role]["output"]
        preview = output[:400] + ("..." if len(output) > 400 else "")
        print(f"\n{role.upper()}:\n{preview}")

    print("\nSummary:", json.dumps(summary, indent=2))
