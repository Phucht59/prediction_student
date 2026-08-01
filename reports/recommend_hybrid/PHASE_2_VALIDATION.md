# recommend_hybrid Phase 2 validation

## Commands and results

| Check | Command | Result |
|---|---|---|
| Phase 1 frozen authority | `.venv-oulad-v2/Scripts/python.exe scripts/recommend_hybrid/validate_phase1_authority.py` | PASS; 30/30 checkpoint files, 75 mappings |
| Targeted unit tests | `.venv-oulad-v2/Scripts/python.exe -m pytest tests/recommend_hybrid -q` | PASS; 30 tests |
| Phase 2 validator | `.venv-oulad-v2/Scripts/python.exe scripts/recommend_hybrid/validate_phase2.py` | `RECOMMEND_HYBRID_PHASE2_FOUNDATION_PASS` |
| Ruff scoped files | `.venv-oulad-v2/Scripts/ruff.exe check src/recommend_hybrid scripts/recommend_hybrid tests/recommend_hybrid` | PASS |

## Gate evidence

- Prediction/logit/probability/class/embedding invariance: exact PASS.
- Five checkpoint hashes and five parameter hashes before/after: unchanged.
- Deterministic eval, 64-D and 32-D shape, architecture/stage lineage: PASS.
- Contract serialization and invalid stage/dimension rejection: PASS.
- Post-cutoff rejection, sensitive-field rejection, missing-not-zero and lineage: PASS.
- Catalog uniqueness/workload/stage/evidence/prerequisite/cycle validation: PASS.
- Candidate generator has no score/rank; `FINAL_EVALUATION` eligible interventions: 0.
- Expert export blinding and blank templates: PASS; fabricated labels: 0.
- Import invalid score and duplicate rejection; raw immutability: PASS.
- Ranker training started: false.

Long validation output is stored under `reports/recommend_hybrid/logs/` and excluded from release tracking by repository policy.
