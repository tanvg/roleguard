"""Verifies RoleGuard text filter correctly identifies and removes non-permitted PHI categories from natural language text."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from langchain_ollama import ChatOllama

from src.pipeline import roleguard_filter_text
from src.roles import CATEGORIES, PERMISSIONS


TEST_TEXT = """
Patient ID: PAT-TEST-001
Diagnosis: Malignant neoplasm of breast (disorder)
Diagnosis code: 254837009
Medication: Tamoxifen 20 MG Oral Tablet
Procedure code: 71651007 Mammography
Lab results: CA 15-3: 28 U/mL
Imaging: Bilateral mammogram with suspicious mass
Department: Oncology
Appointment time: 2020-03-06T22:02:57-05:00
Clinician ID: Dr. Ulysses Friesen
"""


def _check_role(role: str, llm) -> bool:
    print(f"\n--- Role: {role} ---")
    _filtered, audit = roleguard_filter_text(TEST_TEXT, role, llm, source_label=f"test → {role}")

    permitted = PERMISSIONS[role]
    non_permitted = [c for c in CATEGORIES if c not in permitted]
    filtered_out = set(audit.get("categories_filtered_out") or [])
    found = set(audit.get("categories_found") or [])
    passed_through = found - filtered_out

    print(f"  categories_found: {sorted(found)}")
    print(f"  categories_filtered_out (blocked): {sorted(filtered_out)}")
    print(f"  categories passed through: {sorted(passed_through)}")
    print(f"  filtered_text preview:\n{_filtered[:300]}")

    # Non-permitted categories that appear in the text should be blocked.
    expected_blocked = [c for c in non_permitted if c in found]
    missing = [c for c in expected_blocked if c not in filtered_out]
    if missing:
        print(f"  MISSING from filtered_out: {missing}")
        return False

    # Permitted categories that appear should not be filtered out.
    wrongly_blocked = [c for c in found if c in permitted and c in filtered_out]
    if wrongly_blocked:
        print(f"  WRONGLY blocked permitted categories: {wrongly_blocked}")
        return False

    return True


def main() -> int:
    print("=" * 60)
    print("TEST 03: RoleGuard text filter accuracy")
    print("=" * 60)

    llm = ChatOllama(model="llama3")
    scheduling_ok = _check_role("scheduling", llm)
    billing_ok = _check_role("billing", llm)
    passed = scheduling_ok and billing_ok

    print()
    if passed:
        print("PASS: all non-permitted categories blocked for scheduling and billing")
        return 0

    print("FAIL: RoleGuard text filter missed non-permitted categories")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
