# RoleGuard

Privacy-preserving multi-agent LLM pipelines for healthcare. RoleGuard studies how sensitive patient information flows between specialized agents in clinical workflows, measures where privacy leaks occur, and enforces HIPAA-aligned role-based access controls at every inter-agent boundary.

## Project Overview

Healthcare organizations are increasingly adopting multi-agent LLM systems—where a clinical agent, scheduling agent, billing agent, and others collaborate on patient cases via natural language messages. Each agent may only be authorized to see a subset of protected health information (PHI) under HIPAA's minimum necessary standard. Without explicit controls, agents can inadvertently leak PHI across role boundaries during message passing.

**RoleGuard** addresses this problem with two complementary components:

| Component | Purpose |
|-----------|---------|
| **RoleLeak** | A two-tier benchmark that measures privacy leakage at every inter-agent boundary (structured + natural language) |
| **RoleGuard** | Inference-time middleware that filters natural language handoffs using an LLM extract-and-rewrite process based on HIPAA role permissions |

The project runs entirely on local infrastructure: **LangGraph** orchestrates a sequential agent pipeline, **Llama 3** (via **Ollama**) powers inference, and **mCODE STU1** synthetic breast cancer FHIR records provide realistic but non-identifying test cases. All development and evaluation run on an **Ubuntu VM** with **Python 3.12**.

## Architecture

```
 User Query
     │
     ▼
┌──────────────┐
│ Orchestrator │  (LLM decomposes query into subtasks)
└──────┬───────┘
       │
       ▼
┌──────────────┐     FHIR patient data
│   Clinical   │◄──── (RoleGuard structured filter:
│    Agent     │      get_permitted_only)
└──────┬───────┘
       │ clinical_output (natural language)
       ▼
┌──────────────┐
│  RoleGuard   │  extract PHI categories → rewrite permitted-only text
└──────┬───────┘
       │ scheduling_input (dept, time, clinician)
       ▼
┌──────────────┐
│  Scheduling  │
│    Agent     │
└──────┬───────┘
       │ clinical_output + scheduling_output
       ▼
┌──────────────┐
│  RoleGuard   │  extract → rewrite for billing permissions
└──────┬───────┘
       │ billing_input (codes + patient_id)
       ▼
┌──────────────┐
│   Billing    │
│    Agent     │
└──────┬───────┘
       │
       ▼
   Results + RoleLeak measurements
```

**Pipeline flow:**

1. **Orchestrator** (LLM) decomposes the user query into clinical, scheduling, and billing subtasks — no patient data access.
2. **Clinical agent** reads RoleGuard-filtered structured patient data and produces a natural language care coordination summary.
3. **RoleGuard** intercepts clinical output text: extracts PHI categories present, then rewrites the text keeping only categories permitted for scheduling.
4. **Scheduling agent** receives filtered text (department, appointment time, clinician) and produces a calendar entry.
5. **RoleGuard** intercepts combined clinical + scheduling text and rewrites for billing permissions.
6. **Billing agent** receives filtered text (diagnosis/procedure codes + patient ID) and produces an insurance claim.
7. **RoleLeak** measures leakage at each boundary before and after filtering.

### RoleGuard (core contribution)

- Inference-time middleware — no retraining required
- Two-step LLM process: (1) extract PHI categories from text, (2) rewrite text keeping only permitted categories
- Model-agnostic: works with any underlying LLM
- Operates on natural language text, not structured data
- Code categories (`diagnosis_code`, `procedure_code`) keep only bare codes, not free-text labels

### RoleLeak (benchmark)

- **Tier 1:** Deterministic boundary measurement on structured data (received categories vs. `roles.py` permissions)
- **Tier 2:** LLM judge detects PHI categories surfaced in natural language text
- Measures leakage before and after RoleGuard filtering
- Runs in **baseline** mode (no filtering) and **protected** mode (RoleGuard active)

### Two modes

| Mode | Behavior |
|------|----------|
| **Baseline** | No RoleGuard filtering; downstream agents receive full PHI text from upstream outputs |
| **Protected** | RoleGuard active at every inter-agent text boundary |

### Agent trust levels (`src/roles.py`)

- **Clinical** (high trust): diagnosis, diagnosis_code, medication, procedure_code, lab_results, imaging, department, appointment_time, clinician_id, patient_id
- **Scheduling** (medium trust): department, appointment_time, clinician_id, patient_id
- **Billing** (low trust): diagnosis_code, procedure_code, patient_id

## Dataset

- **Source:** mCODE STU1 Synthetic Breast Cancer Records (MITRE Corporation)
- **Breast cancer cohort:** 180 patients verified (178 female + 2 male)
- **Diagnosis:** Malignant neoplasm of breast
- **Procedures:** Chemotherapy, radiation therapy, biopsy, lumpectomy
- **Format:** FHIR R4 JSON (lifetime longitudinal records)
- **Snapshot approach:** most recent oncology-related encounter per patient (not full 10-year history)
- **Download:** https://confluence.hl7.org/display/COD/mCODE+Test+Data

Place extracted FHIR files in `data/scenarios/all_patients/`.

## Setup Instructions

### Prerequisites

- Ubuntu (tested on Ubuntu VM)
- Python 3.12
- [Ollama](https://ollama.com/) installed and running locally
- Sufficient disk space for the Llama 3 model (~5 GB)

### 1. Clone the repository

```bash
git clone <repository-url>
cd roleguard_project
```

### 2. Create and activate the virtual environment

```bash
python3.12 -m venv env
source env/bin/activate
```

### 3. Install Python dependencies

```bash
pip install langgraph langchain-ollama langchain-core python-dotenv pyyaml
```

### 4. Install and pull the Llama 3 model via Ollama

```bash
curl -fsSL https://ollama.com/install.sh | sh
ollama serve &
ollama pull llama3
```

Verify:

```bash
ollama list
# Expected: llama3:latest
```

### 5. Obtain patient data

Download the mCODE STU1 breast cancer dataset from:
https://confluence.hl7.org/display/COD/mCODE+Test+Data

Place extracted FHIR JSON files in `data/scenarios/all_patients/`.

### 6. Verify the environment

```bash
python test.py
```

You should see a one-sentence greeting from Llama 3, confirming Ollama connectivity.

## How to Run

### Quick connectivity test

```bash
source env/bin/activate
python test.py
```

### Inspect role permissions

```bash
python3 src/roles.py
```

### Load and inspect patient snapshots

```bash
python3 src/synthea_loader.py
```

### Run the sequential LangGraph pipeline (baseline + protected)

Runs one breast cancer patient through both modes and prints full message handoffs and RoleLeak violations:

```bash
source env/bin/activate
python3 -u src/pipeline.py
```

### Run RoleLeak Tier 1 (deterministic, all 180 patients)

```bash
python3 -c "
from src.roleleak import run_tier1_only
from src.synthea_loader import load_all_patients
from pathlib import Path
patients = load_all_patients('data/scenarios/all_patients')['breast_cancer']
run_tier1_only(patients, agents={}, llm=None,
               output_file='data/results/roleleak_tier1.jsonl')
"
```

Or via the module `__main__` (Tier 1 all patients, then Tier 1+2 sample):

```bash
python3 src/roleleak.py
```

### Run RoleGuard protected pipeline benchmark

```bash
python3 src/roleguard.py
```

### Generate comparison tables and Pareto curve

```bash
python3 src/evaluate.py
```

Results are written to `data/results/` (JSONL + `evaluation_summary.json`).

> **Note:** Full Tier 2 / protected pipeline runs use many LLM calls and are slow on CPU-only VMs. Start with `src/pipeline.py` (one patient) or Tier 1-only benchmarks.

## Research Questions

1. **How much PHI leaks across inter-agent boundaries in unfiltered multi-agent LLM pipelines?**
   RoleLeak quantifies leakage at each handoff (structured Tier 1 and natural language Tier 2).

2. **Can inference-time role-based text filtering reduce leakage without breaking agent task completion?**
   RoleGuard extract-and-rewrite enforces HIPAA minimum-necessary access on natural language messages.

3. **Which agent roles and boundary types are most prone to privacy leakage?**
   The benchmark profiles leakage by source role, destination role, and message content.

4. **How does local open-weight inference (Llama 3 via Ollama) support privacy-preserving agent design?**
   Running models locally keeps PHI on-premise for healthcare compliance experiments.

5. **Can synthetic mCODE FHIR data support privacy benchmarking of multi-agent clinical workflows?**
   Snapshot oncology encounters provide clean ground truth for RoleLeak evaluation.

## File Structure

```
roleguard_project/
├── README.md              # This file
├── NOTES.md               # Research log (experiments, decisions, open questions)
├── test.py                # Ollama / Llama 3 connectivity smoke test
├── env/                   # Python 3.12 virtual environment (not committed)
│
├── src/
│   ├── pipeline.py        # LangGraph sequential pipeline with orchestrator
│   ├── roleguard.py       # LLM-based text filtering middleware
│   ├── roleleak.py        # Two-tier leakage benchmark
│   ├── roles.py           # HIPAA permission sets
│   ├── synthea_loader.py  # FHIR patient parser
│   └── evaluate.py        # Comparison tables and Pareto analysis
│
├── agents/
│   ├── clinical.py        # Clinical coordination agent
│   ├── scheduling.py      # Scheduling agent
│   └── billing.py         # Billing agent
│
├── data/
│   ├── scenarios/
│   │   └── all_patients/  # mCODE FHIR R4 patient bundles
│   └── results/           # Benchmark outputs and evaluation reports
│
├── cache/                 # LLM response cache (optional)
└── logs/                  # Runtime logs and audit trails
```

## Technology Stack

| Layer | Technology |
|-------|------------|
| Orchestration | [LangGraph](https://github.com/langchain-ai/langgraph) sequential `StateGraph` |
| LLM inference | [Llama 3](https://ollama.com/library/llama3) via [Ollama](https://ollama.com/) |
| LLM bindings | [langchain-ollama](https://python.langchain.com/docs/integrations/chat/ollama/) (`ChatOllama`) |
| Access control | RoleGuard extract-and-rewrite middleware (`src/pipeline.py`, `src/roleguard.py`) |
| Benchmark | RoleLeak Tier 1 + Tier 2 (`src/roleleak.py`) |
| Permissions | HIPAA role matrices (`src/roles.py`) |
| Test data | mCODE STU1 Synthetic Breast Cancer Records (FHIR R4) |
| Data loader | `src/synthea_loader.py` (snapshot oncology encounters) |
| Language | Python 3.12 |
| Platform | Ubuntu VM |

## License

Research use. See repository license file for terms.
