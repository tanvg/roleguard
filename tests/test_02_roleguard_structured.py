"""Verifies RoleGuard's structured dictionary filter eliminates all boundary violations deterministically."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.roleleak import measure_boundary_leakage
from src.roleguard import apply_roleguard_with_audit
from src.roles import PIPELINE
from src.synthea_loader import load_all_patients


def main() -> int:
    print("=" * 60)
    print("TEST 02: RoleGuard structured dictionary filter")
    print("=" * 60)

    patients = load_all_patients("data/scenarios/all_patients")["breast_cancer"][:10]
    print(f"Loaded {len(patients)} breast cancer patients")

    totals = {role: {"violations": 0, "n": 0} for role in PIPELINE}

    for patient in patients:
        for role in PIPELINE:
            filtered, _audit = apply_roleguard_with_audit(patient, role)
            boundary = measure_boundary_leakage(filtered, role)
            totals[role]["n"] += 1
            if boundary["has_violation"]:
                totals[role]["violations"] += 1

    print("\nViolation counts after RoleGuard:")
    for role in PIPELINE:
        print(f"  {role:12} {totals[role]['violations']}/{totals[role]['n']}")

    passed = all(totals[role]["violations"] == 0 for role in PIPELINE)

    print()
    if passed:
        print("PASS: 0% violations for all roles after structured filtering")
        return 0

    print("FAIL: expected 0 violations for all roles after filtering")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
