"""Verifies that without RoleGuard, scheduling and billing agents receive PHI categories outside their permitted scope."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.roleleak import measure_boundary_leakage
from src.roles import PIPELINE
from src.synthea_loader import load_all_patients


def main() -> int:
    print("=" * 60)
    print("TEST 01: Boundary leakage baseline (no RoleGuard)")
    print("=" * 60)

    patients = load_all_patients("data/scenarios/all_patients")["breast_cancer"][:10]
    print(f"Loaded {len(patients)} breast cancer patients")

    totals = {role: {"violations": 0, "n": 0} for role in PIPELINE}

    for patient in patients:
        for role in PIPELINE:
            boundary = measure_boundary_leakage(patient, role)
            totals[role]["n"] += 1
            if boundary["has_violation"]:
                totals[role]["violations"] += 1

    rates = {
        role: (totals[role]["violations"] / totals[role]["n"] * 100 if totals[role]["n"] else 0.0)
        for role in PIPELINE
    }

    print("\nViolation rates:")
    for role in PIPELINE:
        print(
            f"  {role:12} {rates[role]:6.1f}% "
            f"({totals[role]['violations']}/{totals[role]['n']})"
        )

    expected = {"clinical": 0.0, "scheduling": 100.0, "billing": 100.0}
    passed = all(abs(rates[role] - expected[role]) < 0.01 for role in PIPELINE)

    print()
    if passed:
        print("PASS: clinical=0%, scheduling=100%, billing=100%")
        return 0

    print("FAIL: expected clinical=0%, scheduling=100%, billing=100%")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
