# Current Report Registry

This registry separates **current prediction authority** from historical / superseded evidence.

## Current prediction authority (only)

The only current thesis prediction model is **Hybrid CNN–BiLSTM**.

| Topic | Current source |
|---|---|
| Public Hybrid CNN–BiLSTM | `src/prediction/model/hybrid.py` |
| Config | `configs/prediction/hybrid_final.json` |
| Registry | `configs/prediction/registry.json` |
| Canonical report | `reports/prediction/final/FINAL_PREDICTION_MODEL_REPORT.md` |
| Chương 3 | `reports/prediction/final/CHUONG_3.md` |
| Current result tables | `reports/prediction/final/uci_final.csv`, `reports/prediction/final/oulad_final.csv` |
| Information growth | `reports/prediction/final/information_growth.csv` |
| Imbalance / fairness note | `reports/prediction/final/IMBALANCE_AND_FAIRNESS.md` |
| Project map | `PROJECT.md` |
| Decision manifest | `artifacts/prediction/final/FINALIZATION_DECISION.json` |
| Current surface list | `artifacts/prediction/final/CURRENT_SURFACE.json` |

Do not treat any other Hybrid, H1, experiment code-name, thesis_v3 table, or Phase 8 outer CSV as the current prediction authority.

Public names: **Hybrid CNN–BiLSTM**, **Recommendation V**. Public comparison roster: Hybrid / LR / DT / RF / SVM / MLP. XGBoost is not a serving baseline. Outer test not opened.

## Other current subsystems (not the Hybrid prediction model)

| Topic | Current source |
|---|---|
| Recommendation V | `reports/recommend_hybrid/v3/FINAL_RECOMMENDATION_V3_REPORT.md` |
| Live PostgreSQL | `database/live/README.md` |

Non-release research trees (old `artifacts/final`, imbalance experiment, H1, Phase2–4 run dumps) were moved to `test_lab/`. See `test_lab/MOVED_FROM_RELEASE.md`.

## HISTORICAL / SUPERSEDED — DO NOT USE AS CURRENT PREDICTION AUTHORITY

The following are provenance only. They are **not** the current Hybrid CNN–BiLSTM authority.

| Item | Why it is historical |
|---|---|
| `H1_TABULAR_RESIDUAL_EXPERT` and Macro-F1 `0.894…` | Pre-Phase4 OULAD expert; superseded |
| `configs/final/cnn_bilstm_mat.yaml`, `cnn_bilstm_por.yaml`, `h1_tabular_residual_oulad.yaml` | Pre-Phase4 configs |
| `reports/final/thesis_v3/` | Historical thesis result set |
| `reports/prediction/final/uci_table.csv`, `oulad_early_table.csv`, `oulad_final_table.csv` | Phase 8 outer freeze (contains XGB). Kept at original path only because a checksum manifest locks it. Current tables are `uci_final.csv` / `oulad_final.csv`. |
| `artifacts/prediction/final/outer_test_final/`, `bootstrap/`, `development/`, `protocol/`, `recovery/`, `consumption/` | Phase 8 outer / recovery freeze. Checksum-locked paths. Sidecar `HISTORICAL.json` markers sit beside them. |

Phase 8 outer was **not** the Phase 4 evaluation. Phase 4 did not open outer.

Byte-identical extra copies under `artifacts/prediction/historical/phase8/` and `reports/prediction/historical/phase8/` were deleted after SHA match. Checksum-locked originals stay. The index is `artifacts/prediction/final/HISTORICAL_PHASE8.json`.
