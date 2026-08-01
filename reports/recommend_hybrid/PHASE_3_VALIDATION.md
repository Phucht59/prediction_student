# recommend_hybrid Phase 3 validation

## Commands

| Validation | Command | Expected result |
|---|---|---|
| Phase 1 authority | `.venv-oulad-v2/Scripts/python.exe scripts/recommend_hybrid/validate_phase1_authority.py` | `RECOMMEND_HYBRID_PHASE1_AUTHORITY_PASS` |
| Phase 2 regression | `.venv-oulad-v2/Scripts/python.exe -m pytest tests/recommend_hybrid -q` | all Phase 2/3 tests pass |
| Phase 3 policy tests | `.venv-oulad-v2/Scripts/python.exe -m pytest tests/recommend_hybrid/phase3 -q` | 101 targeted tests pass |
| Phase 3 validator | `.venv-oulad-v2/Scripts/python.exe scripts/recommend_hybrid/validate_phase3.py` | `RECOMMEND_HYBRID_PHASE3_POLICY_PASS` |
| Scoped lint | `.venv-oulad-v2/Scripts/ruff.exe check src/recommend_hybrid scripts/recommend_hybrid tests/recommend_hybrid` | PASS |

## Gate summary

- Phase 1 authority and cached Phase 2 prediction invariance: PASS.
- Canonical checkpoints validated against all 30 manifest SHA-256 values; mutation: false.
- UCI MAT/POR configs separate; S0/S1/S2 router and G3 rejection: PASS.
- OULAD arbitrary past-anchor router, pre-20 abstention and final evaluation-only: PASS.
- Future-anchor, post-cutoff, cross-dataset, unsupported-action and missing-evidence misuse counts: 0.
- Risk-only action generation: 0; action suitability score fields: 0.
- Neural action ranker and expert-label dependency: absent.
- Controlled scenarios: 20 UCI + 30 OULAD, all PASS.
- Metamorphic/monotonicity violations: 0.
- Explanation lineage completeness: 100%.
- Deterministic replay: PASS.

Long command output is stored in `reports/recommend_hybrid/logs/phase3_validation.log` and excluded from release tracking.
