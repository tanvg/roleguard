"""HIPAA-grounded role permissions for the multi-agent healthcare pipeline."""

CATEGORIES = [
    "diagnosis",
    "diagnosis_code",
    "medication",
    "procedure_code",
    "lab_results",
    "imaging",
    "department",
    "appointment_time",
    "clinician_id",
    "patient_id",
]

PERMISSIONS = {
    "clinical": {
        "diagnosis",
        "medication",
        "procedure_code",
        "lab_results",
        "imaging",
        "department",
        "clinician_id",
        "patient_id",
    },
    "scheduling": {
        "department",
        "appointment_time",
        "clinician_id",
        "patient_id",
    },
    "billing": {
        "diagnosis_code",
        "procedure_code",
        "patient_id",
    },
}

TRUST_LEVELS = {
    "clinical": "high",
    "scheduling": "medium",
    "billing": "low",
}

PIPELINE = ["clinical", "scheduling", "billing"]


def is_permitted(category: str, agent_role: str) -> bool:
    """Return True if the agent role may access the given PHI category."""
    return category in PERMISSIONS.get(agent_role, set())


def get_violations(received_categories: list[str], agent_role: str) -> list[str]:
    """Return categories received by the agent that exceed its permitted scope."""
    permitted = PERMISSIONS.get(agent_role, set())
    return [cat for cat in received_categories if cat not in permitted]


def get_permitted_only(data: dict, agent_role: str) -> dict:
    """Filter a data dict to only fields the agent role is permitted to see."""
    permitted = PERMISSIONS.get(agent_role, set())
    return {key: value for key, value in data.items() if key in permitted}


def get_permission_summary() -> None:
    """Print all role permissions with checkmarks for allowed, x for denied."""
    for role in PIPELINE:
        trust = TRUST_LEVELS[role]
        print(f"\n{role.upper()} ({trust} trust)")
        for category in CATEGORIES:
            mark = "✓" if is_permitted(category, role) else "✗"
            print(f"  {mark} {category}")


if __name__ == "__main__":
    print("=" * 50)
    print("RoleGuard Permission Matrix")
    print("=" * 50)
    get_permission_summary()

    print("\n" + "=" * 50)
    print("Permission Tests")
    print("=" * 50)

    assert not is_permitted("diagnosis", "billing"), "billing must not see diagnosis"
    print("✓ billing cannot see diagnosis")

    assert is_permitted("diagnosis_code", "billing"), "billing must see diagnosis_code"
    print("✓ billing can see diagnosis_code")

    assert is_permitted("diagnosis", "clinical"), "clinical must see diagnosis"
    print("✓ clinical can see diagnosis")

    sample_data = {
        "diagnosis": "Malignant neoplasm of breast",
        "diagnosis_code": "C50.9",
        "medication": "Tamoxifen",
        "procedure_code": "77401",
        "lab_results": "CA 15-3: 28 U/mL",
        "imaging": "Mammogram: suspicious mass",
        "department": "Oncology",
        "appointment_time": "2026-03-15 10:00",
        "clinician_id": "DR-1042",
        "patient_id": "PAT-001",
    }

    print("\n" + "=" * 50)
    print("Filtered Data Example (billing role)")
    print("=" * 50)
    filtered = get_permitted_only(sample_data, "billing")
    for key, value in filtered.items():
        print(f"  {key}: {value}")

    violations = get_violations(list(sample_data.keys()), "billing")
    print(f"\nViolations if billing received full record: {violations}")
