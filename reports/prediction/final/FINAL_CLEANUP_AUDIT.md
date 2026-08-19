# Final cleanup audit (4 issues)

## 1. Issue

Active-looking `reports/prediction/final` and `artifacts/prediction/final` mixed Phase 4 authority with Phase 8 XGB/outer freeze. `CURRENT_REPORTS.md` still called H1 the OULAD authority. Active `src/prediction` could not build OULAD weekly tensors from raw tables. Overfit audit copied one dataset-level gap/std onto every stage.

## 2. Root cause

Finalization promoted C0 but left checksum-locked Phase 8 files in the same folders, reused a stale report registry, wrapped arrays instead of owning the Phase 4 builder, and stamped macro overfit stats onto stages.

## 3. Files changed

- Historical copies: `artifacts/prediction/historical/phase8/`, `reports/prediction/historical/phase8/`
- Markers + mapping: `CURRENT_SURFACE.json`, `HISTORICAL_PHASE8.json`, `PATH_MIGRATION.json`, `*/HISTORICAL.json`
- `reports/CURRENT_REPORTS.md` rewritten
- `src/prediction/data/oulad_features.py` — Phase 4 cutoff-safe builder (no `experiments/` import)
- `OVERFIT_AUDIT.json` recomputed per stage from official 3×3 runs
- Tests in `tests/prediction/test_phase4_authority.py`
- README pipeline wording

Checksum-locked Phase 8 files stay at original paths so `PREDICTION_PHASE8_MIGRATION_MANIFEST.csv` remains valid.

## 4. Validation

```text
python project.py prediction status / registry / validate
pytest tests/prediction tests/hybrid_vnext
```

Canonical `uci_final.csv` / `oulad_final.csv` contain only Hybrid, LR, DT, RF, SVM, MLP.

## 5. Scientific impact

```text
NO MODEL RETRAINING
NO HPO
NO OUTER EVALUATION
NO METRIC CHANGE
NO ARCHITECTURE CHANGE
```

UCI S0 generalization gap is independently larger than S1 and S2 (0.125 vs 0.035 / 0.020). That is an audit correction, not a change to reported PR-AUC.
