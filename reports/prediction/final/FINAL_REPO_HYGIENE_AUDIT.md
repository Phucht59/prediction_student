# Final repository hygiene audit

## 1. Current authority

```text
model_id = hybrid
display_name = Hybrid
public_class = Hybrid
architecture = Hybrid CNN-BiLSTM
source_phase = Phase4
evaluation_status = robust_inner_finalized
outer_test_used = false
```

UCI S0/S1/S2 and OULAD 20/35/50/75/100 are information states of one fitted model per dataset. Active comparators: LR / DT / RF / SVM / MLP. XGBoost is historical only.

## 2. Files removed

- Root duplicates: `HYBRID_VNEXT_PHASE2_DESIGN_REPORT.md`, `HYBRID_VNEXT_PHASE3_FINAL_REPORT.md`, `HYBRID_VNEXT_PHASE4_FINAL_REPORT.md` (byte-identical to `reports/hybrid_vnext/phase*/`)
- Duplicate Phase 8 trees: `artifacts/prediction/historical/phase8/`, `reports/prediction/historical/phase8/`
- Redundant active alias: `src/prediction/data/final100.py`
- Untracked zero-byte scratch logs (gitignored): `artifacts/audit/phase3/logs/supervisor.stdout.log`, `artifacts/prediction/reconstructed/reconstruction_stdout.log`, `artifacts/prediction/reconstructed/reconstruction_stderr_3.log`

## 3. Files moved/archived

- Legacy `PROJECT.md` → `reports/prediction/historical/pre_phase4/PROJECT_LEGACY.md`
- `configs/prediction/hybrid_phase8.json` → `configs/prediction/historical/hybrid_phase8.json`

## 4. Duplicate files eliminated

Phase 2/3/4 reports now have one canonical copy under `reports/hybrid_vnext/`.
Phase 8 checksum-locked originals stay at original paths with sidecar `HISTORICAL.json` markers. Extra SHA-identical copies were deleted. `PATH_MIGRATION.json` records that deletion.

## 5. PROJECT.md correction

Root `PROJECT.md` is the Phase 4 binary Hybrid map. Multiclass / `cnn_bilstm_*` / old outer claims exist only in the legacy archive.

## 6. Environment cleanup

- `environment.yml`: no `xgboost`, no `optuna`
- `environment.research.yml`: Optuna + XGBoost for Phase 2/3/4 HPO and historical XGB tables
- Active `src/prediction` and `tests/` do not import xgboost or optuna

## 7. OULAD raw→full-input reproducibility

`fit_oulad_preprocessor` + required `build_oulad_information_state(..., preprocessor=)` emit non-zero static, 11 temporal channels, 13 aggregates, FIT-only scaling, cutoff `observation_start <= t < cutoff`, and a Hybrid forward smoke test. Zero-width static is no longer a silent default for the high-level builder.

## 8. final100/config cleanup

`100pct` is `canonical_oulad_state`. No separate FINAL100 module. Active configs: `hybrid_final.json`, `baselines_final.json`, `registry.json`.

## 9. Validation results

```text
python project.py prediction status   : THESIS_FINAL / C0 / outer_test_used=false
python project.py prediction registry : xgboost_active=false; fitted_instances=[uci, oulad]
python project.py prediction validate : PASS
pytest tests/prediction + tests/hybrid_vnext : 24 passed
canonical metric SHA unchanged
Phase4 report SHA unchanged
```

Canonical hashes:

```text
reports/prediction/final/uci_final.csv
  7db7ef3c61c8be3252b0a2063d4ca85c254d90aca9c53d74f7c92a1b6dda10a5
reports/prediction/final/oulad_final.csv
  5e63e6ee017abca952bd850eb15dbcf18a19be51757cd2ad58072f83abc60d83
reports/prediction/final/information_growth.csv
  b6c812affb0b8ee8c70acea808544385eb3764abb46c4358a051973c41c57b77
artifacts/hybrid_vnext/phase4/robust_summary.json
  345ccdaa7f4f0b55481273704bd8eca978207d215f1bc8587b11e934389c9bc4
artifacts/hybrid_vnext/phase4/BASELINE_CEILING.json
  87eff46016ad92a0f93ea248610e1a23fe5d2b54fbef126c8e42635d569d441a
```

Clutter scan: no tracked `*.tmp`/`*.bak`/`*.old`/`*.orig`; no Optuna sqlite in the tree. Empty historical checkpoint folders and recommendation package directories were left in place (not SAFE_TO_DELETE as current-authority clutter).

## 10. Scientific invariants

```text
MODEL_CHANGED = false
TRAINING_PERFORMED = false
HPO_PERFORMED = false
OUTER_OPENED = false
METRICS_CHANGED = false
RECOMMENDATION_CHANGED = false
DATABASE_CHANGED = false
```
