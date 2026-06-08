# RoleGuard

Privacy-preserving multi-agent LLM pipelines for healthcare. RoleGuard studies how sensitive patient information flows between specialized agents in clinical workflows, measures where privacy leaks occur, and enforces HIPAA-aligned role-based access controls at every inter-agent boundary.

## Project Overview

Healthcare organizations are increasingly adopting multi-agent LLM systems—where a clinical retrieval agent, scheduling agent, billing agent, and others collaborate on patient cases. Each agent may only be authorized to see a subset of protected health information (PHI) under HIPAA's minimum necessary standard. Without explicit controls, agents can inadvertently leak PHI across role boundaries during message passing.

**RoleGuard** addresses this problem with two complementary components:

| Component | Purpose |
|-----------|---------|
| **RoleLeak** | A benchmark that measures privacy leakage at every inter-agent boundary in a LangGraph pipeline |
| **RoleGuard** | Middleware that filters inter-agent messages based on HIPAA role permissions before they reach the next agent |

The project runs entirely on local infrastructure: **LangGraph** orchestrates the agent pipeline, **Llama 3** (via **Ollama**) powers inference, and **Synthea** synthetic patient data provides realistic but non-identifying test cases. All development and evaluation run on an **Ubuntu VM** with **Python 3.12**.

## Architecture

```
                    ┌─────────────────────────┐
                    │   Patient Data (FHIR)   │
                    │  mCODE Breast Cancer    │
                    └───────────┬─────────────┘
                                │
                                ▼
                    ┌─────────────────────────┐
                    │       RoleGuard         │
                    │  (HIPAA role filter)    │
                    └───────────┬─────────────┘
                                │
            ┌───────────────────┼───────────────────┐
            │                   │                   │
            ▼                   ▼                   ▼
    ┌───────────────┐   ┌───────────────┐   ┌───────────────┐
    │ Clinical Agent│   │Scheduling Agent│   │ Billing Agent │
    │ (high trust)  │   │ (medium trust) │   │ (low trust)   │
    └───────┬───────┘   └───────┬───────┘   └───────┬───────┘
            │                   │                   │
            └───────────────────┼───────────────────┘
                                │
                                ▼
                    ┌─────────────────────────┐
                    │        RoleLeak         │
                    │  measures received vs.  │
                    │   permitted per agent   │
                    └─────────────────────────┘
```

**Pipeline flow:**

- Each agent requests information from RoleGuard
- RoleGuard checks the requesting agent's role
- RoleGuard returns ONLY permitted information to that agent
- **Clinical agent** (high trust): sees diagnosis, medication, procedures, lab results
- **Scheduling agent** (medium trust): sees only department, appointment time, clinician id
- **Billing agent** (low trust): sees only procedure code, insurance id
- **RoleLeak** measures what each agent receives vs what it should get

## Dataset

- **Source:** mCODE STU1 Synthetic Breast Cancer Records (MITRE Corporation)
- **Total patients:** 196 (180 female, 16 male)
- **Diagnosis:** Malignant neoplasm of breast
- **Procedures:** Chemotherapy, radiation therapy, biopsy, lumpectomy
- **Format:** FHIR R4 JSON
- **Download:** https://confluence.hl7.org/display/COD/mCODE+Test+Data

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

> The project may also use `langchain-groq` for optional cloud-based comparison experiments. Install only what your configuration requires.

### 4. Install and pull the Llama 3 model via Ollama

```bash
# Install Ollama (if not already installed)
curl -fsSL https://ollama.com/install.sh | sh

# Start the Ollama service
ollama serve &

# Pull the model
ollama pull llama3
```

Verify the model is available:

```bash
ollama list
# Expected: llama3:latest
```

### 5. Obtain the Synthea synthetic patient data

Download or generate Synthea FHIR/CSV exports and place scenario files in `data/scenarios/`. Synthea produces fully synthetic patients with no real PHI, making it safe for privacy research.

- Synthea project: https://github.com/synthetichealth/synthea
- Place one scenario per file (e.g., `data/scenarios/patient_001.json`)

For this project:
Download the mCODE STU1 breast cancer dataset from:
https://confluence.hl7.org/display/COD/mCODE+Test+Data

Place the extracted FHIR files in `data/scenarios/all_patients/`

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

### Run the RoleLeak benchmark (baseline — no filtering)

Measure privacy leakage across all inter-agent boundaries without RoleGuard middleware:

```bash
source env/bin/activate
python -m src.roleleak --scenario data/scenarios/<patient_file> --output data/results/
```

### Run the pipeline with RoleGuard filtering

Execute the full privacy-preserving pipeline with HIPAA role enforcement at each boundary:

```bash
source env/bin/activate
python -m src.pipeline --scenario data/scenarios/<patient_file> --guard enabled --output data/results/
```

### Compare filtered vs. unfiltered leakage

Run both modes on the same scenario set to quantify RoleGuard's reduction in PHI leakage:

```bash
source env/bin/activate
python -m src.evaluate --scenarios data/scenarios/ --output data/results/comparison.json
```

> **Note:** Pipeline entry points (`src/roleleak`, `src/pipeline`, `src/evaluate`) are the intended interfaces. Adjust commands to match the modules as they are implemented.

## Research Questions

This project investigates the following questions:

1. **How much PHI leaks across inter-agent boundaries in unfiltered multi-agent LLM pipelines?**
   RoleLeak quantifies leakage at each handoff, broken down by PHI category (demographics, diagnoses, medications, billing codes, etc.).

2. **Can middleware role-based filtering reduce leakage without breaking clinical task performance?**
   RoleGuard enforces HIPAA minimum-necessary access. We measure the trade-off between privacy protection and downstream agent accuracy.

3. **Which agent roles and boundary types are most prone to privacy leakage?**
   The benchmark profiles leakage by source role, destination role, and message type to identify high-risk handoffs.

4. **How does local open-weight inference (Llama 3 via Ollama) compare to cloud APIs in privacy-preserving agent design?**
   Running models locally keeps PHI on-premise, supporting compliance requirements for healthcare deployments.

5. **Can synthetic patient data (Synthea) reliably substitute for real EHR data in privacy benchmarking?**
   We evaluate whether Synthea scenarios produce leakage patterns representative of real multi-agent clinical workflows.

## File Structure

```
roleguard_project/
├── README.md              # This file
├── NOTES.md               # Research log (experiments, decisions, open questions)
├── test.py                # Ollama / Llama 3 connectivity smoke test
├── env/                   # Python 3.12 virtual environment (not committed)
│
├── src/                   # Core library and pipeline entry points
│   ├── pipeline.py        # LangGraph multi-agent healthcare pipeline
│   ├── roleguard.py       # HIPAA role-based message filter middleware
│   ├── roleleak.py        # Privacy leakage benchmark and scoring
│   ├── roles.py           # HIPAA role definitions and permission matrices
│   └── evaluate.py        # Comparative evaluation (filtered vs. unfiltered)
│
├── agents/                # Agent definitions and system prompts
│   ├── clinical.py        # Clinical retrieval agent (high trust)
│   ├── scheduling.py      # Scheduling agent (medium trust)
│   └── billing.py         # Billing / administrative agent (low trust)
│
├── data/
│   ├── scenarios/         # Synthea synthetic patient scenario files
│   └── results/           # Benchmark outputs and evaluation reports
│
├── cache/                 # LLM response cache (optional, for reproducibility)
└── logs/                  # Runtime logs and audit trails
```

## Technology Stack

| Layer | Technology |
|-------|------------|
| Orchestration | [LangGraph](https://github.com/langchain-ai/langgraph) |
| LLM inference | [Llama 3](https://ollama.com/library/llama3) via [Ollama](https://ollama.com/) |
| LLM bindings | [langchain-ollama](https://python.langchain.com/docs/integrations/chat/ollama/) |
| Test data | [Synthea](https://github.com/synthetichealth/synthea) synthetic patients |
| Language | Python 3.12 |
| Platform | Ubuntu VM |

## License

Research use. See repository license file for terms.
