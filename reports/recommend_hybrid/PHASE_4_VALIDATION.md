# Phase 4 validation

## Commands

```text
python -m pytest tests/recommend_hybrid/phase4 -q
python -m pytest tests/recommend_hybrid/phase3/test_shared_policy.py tests/recommend_hybrid/phase3/test_uci_policy.py tests/recommend_hybrid/phase3/test_oulad_policy.py -q
python scripts/recommend_hybrid/validate_phase4.py
ruff check <Phase 4 Python files>
git diff --check
```

## Validated state

- Targeted Phase 4 tests: 31 passed, 0 failed.
- Phase 3 routing/policy regressions: 101 passed, 0 failed.
- Phase 4 validator: `RECOMMEND_HYBRID_PHASE4_PLANNING_PASS`.
- Canonical checkpoints: 30/30 SHA-256 valid; no byte mutation.
- Prediction baseline: unchanged.
- Constraint violations: 0 across every declared category.
- ABSTAIN and EVALUATION_ONLY action count: 0.
- Explanation/evidence lineage completeness: 100%.
- Persistence round-trip and deterministic replay: PASS.
- Legacy sentinel data: unchanged.

The validator uses small deterministic fixtures and performs no training, full-dataset inference, benchmark, database reload, or background service execution.
