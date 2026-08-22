# Lock — Hybrid C0 prediction + Recommendation V

Frozen `2026-08-22T07:21:20Z`. Outer test: **false**.

## What is locked on main

| Surface | Identity | Path |
|---|---|---|
| Public prediction | Hybrid CNN–BiLSTM, architecture **C0**, `model_id=hybrid` | `src/prediction/` |
| Config | one fitted model per dataset, all information states | `configs/prediction/hybrid_final.json` |
| Recommendation | Recommendation V, consumes `PredictionResult` only | `src/recommend_hybrid/v3/` |
| Rec adapter | no CNN/LSTM inspection | `src/recommend_hybrid/prediction_adapter.py` |

Recommendation does not refit Hybrid. OULAD `100pct` cannot map to an intervention. Rec compatibility check: **PASS**, no rec code change.

## Four headline metrics

Accuracy, Precision (risk class), F1, and the highest of AP vs ROC-AUC (**ROC-AUC** on this lock). STOP threshold only.

### Serving Hybrid C0 (Phase4 inner 3×3) — public tables

From `reports/prediction/final/uci_final.csv` and `oulad_final.csv`.

**UCI**

| Mốc | Accuracy | Precision | F1 | Cao nhất (PR-AUC) |
|---|---:|---:|---:|---:|
| S0 | 0.5213 | 0.2911 | 0.4291 | 0.4547 |
| S1 | 0.8553 | 0.6604 | 0.6899 | 0.8214 |
| S2 | 0.9094 | 0.7654 | 0.8010 | 0.9101 |

**OULAD**

| Mốc | Accuracy | Precision | F1 | Cao nhất (PR-AUC) |
|---|---:|---:|---:|---:|
| 20% | 0.6862 | 0.6033 | 0.6781 | 0.7624 |
| 35% | 0.7435 | 0.6613 | 0.7001 | 0.8058 |
| 50% | 0.8001 | 0.7445 | 0.7306 | 0.8483 |
| 75% | 0.8628 | 0.8516 | 0.7807 | 0.8885 |
| 100% | 0.9034 | 0.9048 | 0.8372 | 0.9204 |

### Research C0-R vs one-weight baselines (this campaign)

Full tables: `reports/research/hybrid_superiority_v2/HEADLINE_FOUR_METRICS.md`.

**C0-R UCI**

| Mốc | Accuracy | Precision | F1 | ROC-AUC |
|---|---:|---:|---:|---:|
| S0 | 0.6796 | 0.3846 | 0.4149 | 0.6957 |
| S1 | 0.8655 | 0.7240 | 0.7180 | 0.9381 |
| S2 | 0.9031 | 0.7826 | 0.7975 | 0.9699 |

**C0-R OULAD**

| Mốc | Accuracy | Precision | F1 | ROC-AUC |
|---|---:|---:|---:|---:|
| 20% | 0.6811 | 0.5991 | 0.6741 | 0.7773 |
| 35% | 0.7498 | 0.6745 | 0.7008 | 0.8308 |
| 50% | 0.8062 | 0.7675 | 0.7324 | 0.8771 |
| 75% | 0.8705 | 0.8649 | 0.7919 | 0.9168 |
| 100% | 0.9051 | 0.9279 | 0.8366 | 0.9370 |

Fair AP gate: UCI **pass**, OULAD **fail** (Δ AP −0.001 to −0.004 on warm vs one-weight XGB/CatBoost). Defense status remains **NOT_READY_FOR_DEFENSE**. No serving cutover of weights; public class stays C0.

## Not opened

Outer test. Rec Panel B. Gemini in prediction HPO.
