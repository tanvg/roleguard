"""
Interactive demo: you play the billing agent.
Type questions to clinical agent and see what RoleGuard allows through.
"""
import warnings
warnings.filterwarnings('ignore')
from langchain_ollama import ChatOllama
from src.synthea_loader import load_all_patients
from src.pipeline import roleguard_filter_text
from src.roles import PERMISSIONS

llm = ChatOllama(model='llama3')
patients = load_all_patients('data/scenarios/all_patients')['breast_cancer']
patient = patients[0]

print("=" * 60)
print("RoleGuard Interactive Demo")
print("=" * 60)
print(f"Patient: {patient['patient_id']}")
print(f"You are the BILLING AGENT")
print(f"Your permitted categories: {PERMISSIONS['billing']}")
print("=" * 60)
print("Type a question to send to the clinical agent.")
print("RoleGuard will filter the response before you see it.")
print("Type 'quit' to exit.")
print()

while True:
    question = input("BILLING AGENT > ").strip()
    
    if question.lower() == 'quit':
        break
    if not question:
        continue
    
    print("\nAsking clinical agent...")
    
    clinical_prompt = f"""You are a clinical agent with full access 
to this patient's record:

Patient ID: {patient['patient_id']}
Diagnosis: {patient['diagnosis']} (code: {patient['diagnosis_code']})
Medications: {', '.join(patient['medication'])}
Procedures: {', '.join(patient['procedure_code'][:3])}
Department: {patient['department']}
Appointment: {patient['appointment_time']}
Clinician: {patient['clinician_id']}

The billing agent asks: {question}

Answer helpfully with full clinical details."""

    clinical_response = llm.invoke(clinical_prompt).content
    
    print("\n--- Clinical Agent said (BEFORE RoleGuard) ---")
    print(clinical_response)
    
    print("\n--- RoleGuard filtering for billing... ---")
    filtered, audit = roleguard_filter_text(
        clinical_response,
        'billing',
        llm,
        source_label='clinical → billing (clarification)'
    )
    
    print("\n--- What YOU (billing) actually receive ---")
    print(filtered)
    print()
    print(f"Blocked: {audit['categories_filtered_out']}")
    print(f"Filter rate: {audit['filter_rate']:.0%} removed")
    print()
