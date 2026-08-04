"""Run all RoleGuard tests in order and print a final PASS/FAIL summary."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

TESTS = [
    "test_01_boundary_leakage_baseline.py",
    "test_02_roleguard_structured.py",
    "test_03_roleguard_accuracy.py",
    "test_04_clarification_loop.py",
    "test_05_adversarial_request.py",
    "test_06_boundary_analysis.py",
    "test_07_indirect_extraction.py",
    "test_08_roleplay_bypass.py",
    "test_09_partial_info_fishing.py",
    "test_10_scheduling_extraction.py",
    "test_11_inference_attack.py",
]


def main() -> int:
    tests_dir = Path(__file__).resolve().parent
    results: list[tuple[str, bool, int]] = []

    print("=" * 60)
    print("RoleGuard Test Suite")
    print("=" * 60)

    for name in TESTS:
        path = tests_dir / name
        print(f"\n>>> Running {name} ...", flush=True)
        completed = subprocess.run(
            [sys.executable, "-u", str(path)],
            cwd=str(tests_dir.parent),
        )
        passed = completed.returncode == 0
        results.append((name, passed, completed.returncode))
        print(f"<<< {name}: {'PASS' if passed else 'FAIL'} (exit={completed.returncode})", flush=True)

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"{'Test':<45} {'Result':>6}")
    print("-" * 55)
    for name, passed, _code in results:
        print(f"{name:<45} {'PASS' if passed else 'FAIL':>6}")

    all_passed = all(passed for _name, passed, _code in results)
    print("-" * 55)
    print(f"Overall: {'PASS' if all_passed else 'FAIL'}")
    return 0 if all_passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
