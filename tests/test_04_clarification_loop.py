"""Verifies the billing-to-clinical clarification loop activates when needed and RoleGuard filters the return path."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from langchain_ollama import ChatOllama

from src.pipeline import run_pipeline
from src.synthea_loader import load_all_patients

TARGET_PATIENT_ID = "1c83fb03-d139-4626-bb39-4a062c22b533"


def main() -> int:
    print("=" * 60)
    print("TEST 04: Clarification loop")
    print("=" * 60)

    patients = load_all_patients("data/scenarios/all_patients")["breast_cancer"]
    patient = next((p for p in patients if p.get("patient_id") == TARGET_PATIENT_ID), None)
    if patient is None:
        print(f"FAIL: patient {TARGET_PATIENT_ID} not found")
        return 1

    print(f"Loaded patient {TARGET_PATIENT_ID}")
    llm = ChatOllama(model="llama3")
    print("Running protected pipeline...")
    state = run_pipeline(patient, llm, baseline_mode=False)

    rounds = int(state.get("clarification_round") or 0)
    request = state.get("clarification_request") or ""
    response = state.get("clarification_response") or ""
    billing_input = state.get("billing_input") or ""

    print(f"\nclarification_round: {rounds}")
    print(f"clarification_request:\n{request}")
    print(f"\nclarification_response:\n{response}")
    print(f"\nbilling_input (truncated):\n{billing_input[:400]}")

    clarification_audits = [
        a
        for a in (state.get("roleguard_audits") or [])
        if "clarification" in str(a.get("boundary", "")).lower()
    ]
    print(f"\nclarification-related audits: {len(clarification_audits)}")
    for audit in clarification_audits:
        print(
            f"  boundary={audit.get('boundary')} "
            f"filtered_out={audit.get('categories_filtered_out')} "
            f"baseline={audit.get('baseline_mode')}"
        )

    triggered = rounds > 0 and bool(request)
    filtered = bool(response) and (
        len(clarification_audits) > 0
        or any(
            "clarification" in str(m.get("boundary", "")).lower()
            for m in (state.get("messages") or [])
        )
    )

    print()
    if triggered and filtered:
        print("PASS: clarification triggered and response was filtered")
        return 0

    print(
        "FAIL: expected clarification_round > 0 and a filtered clarification_response "
        f"(triggered={triggered}, filtered={filtered})"
    )
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
