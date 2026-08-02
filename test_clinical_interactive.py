import warnings
warnings.filterwarnings('ignore')
from langchain_ollama import ChatOllama
from src.synthea_loader import load_all_patients
from src.pipeline import roleguard_filter_text
from src.roles import PERMISSIONS

llm = ChatOllama(model='llama3')
patients = load_all_patients('data/scenarios/all_patients')['breast_cancer']
idx = 0
patient = patients[idx]
roleguard_enabled = True

def get_clinical_response(q, p):
    prompt = f"""You are a senior clinical agent with full EHR access.
Patient ID: {p['patient_id']}
Diagnosis: {p['diagnosis']} (code: {p['diagnosis_code']})
Medications: {', '.join(p['medication'])}
Procedures: {', '.join(p['procedure_code'][:3])}
Lab Results: {', '.join(p['lab_results'][:3]) if p['lab_results'] else 'None'}
Department: {p['department']}
Appointment: {p['appointment_time']}
Clinician: {p['clinician_id']}
Clinical agent asks: {q}
Answer with full clinical detail."""
    return llm.invoke(prompt).content

while True:
    status = "ON" if roleguard_enabled else "OFF"
    print(f"\n[RoleGuard {status}] Patient: {patient['patient_id'][:8]}...")
    print(f"You are the CLINICAL AGENT")
    print(f"Permitted: {PERMISSIONS['clinical']}")
    print("1. Ask question")
    print("2. Toggle RoleGuard ON/OFF")
    print("3. Status")
    print("4. Switch patient")
    print("5. Quit")
    choice = input("Choose (1-5) > ").strip()

    if choice == '5':
        print("Goodbye!")
        break
    elif choice == '1':
        q = input("Your question > ").strip()
        if not q:
            continue
        print("Asking senior clinical agent...")
        response = get_clinical_response(q, patient)
        if roleguard_enabled:
            filtered, audit = roleguard_filter_text(
                response, 'clinical', llm, 'data source → clinical')
            print("\n--- You receive (RoleGuard ON) ---")
            print(filtered)
            print(f"Blocked: {audit['categories_filtered_out']}")
            print(f"Filter rate: {audit['filter_rate']:.0%} removed")
        else:
            print("\n--- You receive (RoleGuard OFF) ---")
            print(response)
            print("\n⚠ WARNING: Full PHI exposed")
    elif choice == '2':
        roleguard_enabled = not roleguard_enabled
        print(f"RoleGuard is now {'ENABLED' if roleguard_enabled else 'DISABLED'}")
    elif choice == '3':
        print(f"RoleGuard : {status}")
        print(f"Patient   : {patient['patient_id']}")
        print(f"Diagnosis : {patient['diagnosis']}")
        print(f"Permitted : {PERMISSIONS['clinical']}")
    elif choice == '4':
        idx = (idx + 1) % len(patients)
        patient = patients[idx]
        print(f"Switched to patient {idx}: {patient['patient_id']}")
    else:
        print("Invalid — enter 1-5")
