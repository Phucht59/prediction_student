# Final release validation

## Gate result

`PHASE_5_PASS` after `RECOMMEND_HYBRID_PHASE5_FINAL_PASS`.

## Locked prerequisites

Phase 1 authority, Phase 2 prediction invariance, Phase 3 policy manifest and Phase 4 planning validation are PASS. Architecture hash remains `df5cd885b96e5cea4b840bfc5ca59c08c095f5887df8dd8dcef738edfe8bf70e`; parameter count remains 160,492. Prediction baseline and checkpoint bytes are unchanged. Current policy hashes match the Phase 3 manifest and planning configuration hash is recorded in the final release manifest.

## Evaluation checks

- UCI MAT/POR S0, S1 and S2: evaluated.
- OULAD canonical anchors plus inter-stage 25/36/63/76: evaluated.
- Safety and constraint violations: 0 in every declared category.
- Evidence support and explanation lineage: 100%.
- Scenario and metamorphic pass rates: 100%.
- Monotonicity and uncertainty-safety violations: 0.
- Controlled robustness: 7/7 PASS.
- Deterministic replay and plan-hash match: 100%.
- Four-variant ablation: valid; official policy unchanged.
- Student-level bootstrap: 1,000 replicates with fixed seed 20260801.
- Scientific claim matrix, model card and thesis report: present and boundary-complete.

## Test commands

```text
python -m pytest tests/recommend_hybrid/phase5 -q
python -m pytest <essential Phase 3-4 regression subset> -q
python scripts/recommend_hybrid/validate_phase5.py
ruff check scripts/recommend_hybrid/validate_phase5.py tests/recommend_hybrid/phase5
git diff --check
```

The final targeted invocation reports 43 passed and 0 failed: 12 Phase 5 evaluation tests plus 31 essential Phase 3–4 regression tests. Locked prior-phase evidence records 101 Phase 3 policy tests and 31 Phase 4 tests passing.

Final test counts are recorded in `FINAL_RELEASE_MANIFEST.json`. Evaluation used no training, fine-tuning, outcome-guided threshold selection, expert/user simulation, causal model, full benchmark rerun or database mutation.
