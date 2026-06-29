# RoleGuard Research Notes

## Daily Log

### June 8 2026
- Set up Ubuntu VM (16 cores, 31GB RAM, 191GB storage)
- Installed Python 3.12, Ollama, Llama3 (CPU only mode, no GPU)
- Connected Cursor to VM via SSH (tg-vm host)
- Created project structure with src/, agents/, data/, cache/, logs/
- Connected to GitHub at github.com/tanvg/roleguard
- Installed Java 21 and Synthea
- Downloaded mCODE STU1 breast cancer dataset (196 patients - 180 female, 16 male)
- Combined female/male/assorted into single dataset at data/scenarios/all_patients/
- Updated README with correct architecture, dataset section, pipeline flow
- Decided on core pipeline: Clinical Agent → Scheduling Agent → Billing Agent
- RoleGuard filters based on requesting agent's role, not outgoing messages
- Dataset: breast cancer patients with chemotherapy, radiation, biopsy procedures

## Decisions

### Decision: Snapshot approach for synthea_loader.py
- Each patient scenario = most recent Encounter (the "current visit")
- Diagnosis/medication/procedures associated with that recent period
- Rationale: matches paper's Jane Doe example, gives clean ground truth 
  for RoleLeak evaluation, avoids ambiguous violation counting across 
  10-year history
- Future work: extend to longitudinal/multi-visit leakage tracking

## Open Questions
- How many patients to use for final benchmark? (have 196 total)
- Should I test with multiple models later?
- What strictness levels for RoleGuard evaluation?
- Add diabetes patients later for Phase 2 cross-department leakage?

## Issues & Solutions
- Synthea -m breast_cancer flag doesn't guarantee all patients have cancer
  Solution: Used mCODE STU1 pre-generated dataset instead
- Git push failed with password authentication
  Solution: Used personal access token embedded in remote URL
- Cursor SSH couldn't find VM
  Solution: Added tg-vm host to ~/.ssh/config manually

## Ideas
- Phase 2: add diabetes clinical agent, show cross-department leakage compounds
- Test RoleGuard with different prompt strictness levels for Pareto curve

## Questions and Advise needed on
- Is 196 patients enough for a statistically significant benchmark?
- Should evaluation section compare against any existing baselines?
- Which venue to target for submission?
