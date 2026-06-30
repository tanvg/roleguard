"""Scheduling agent for the RoleGuard multi-agent pipeline."""

from __future__ import annotations

import json


def scheduling_agent(patient_data: dict, llm) -> dict:
    """Generate a calendar appointment entry from the provided patient data."""
    prompt = (
        "You are a scheduling agent. Given the following information, "
        "produce a calendar appointment entry (department, time, clinician).\n\n"
        f"Patient information:\n{json.dumps(patient_data, indent=2)}"
    )
    response = llm.invoke(prompt)
    return {
        "agent": "scheduling",
        "received_data": patient_data,
        "output": response.content,
    }


if __name__ == "__main__":
    import sys
    import warnings
    from pathlib import Path

    from langchain_ollama import ChatOllama

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    warnings.filterwarnings("ignore")

    from src.synthea_loader import load_all_patients

    llm = ChatOllama(model="llama3")
    data_dir = Path(__file__).resolve().parent.parent / "data" / "scenarios" / "all_patients"
    patient = load_all_patients(str(data_dir))["breast_cancer"][0]
    result = scheduling_agent(patient, llm)
    print(result["output"])
