# Dự đoán rủi ro học tập sinh viên

Thesis-final prediction authority is **Phase 4 Hybrid C0**.

## Final model

- Public name: `Hybrid` (`model_id = hybrid`, class `Hybrid`)
- Architecture: **C0** — parallel CNN ∥ BiLSTM, corrected availability, 3-way masked softmax, one binary risk logit
- Shared structural widths: `d_fuse=128`, `cnn_channels=64`, `bilstm_hidden=128`
- UCI and OULAD use the **same** Hybrid architecture

They differ only because input dimensions and data semantics differ.

### Information states of one fitted model

| Dataset | States of the same checkpoint |
|---|---|
| UCI | S0 → S1 → S2 |
| OULAD | 20% → 35% → 50% → 75% → 100% |

There is no separate OULAD-100 model and no dataset-specific topology.

## Active baselines

`LR / DT / RF / SVM / MLP`

XGBoost is **not** an active comparator in the final project scope. Historical XGBoost numbers remain in older archives only.

## Final evaluation status

- Thesis-final numbers are the **Phase 4 robust inner 3×3** protocol.
- **Outer was not opened** during Phase 4 or this finalization.
- The research gate recorded `NOT_READY_FOR_FINAL_EVAL` because UCI did not beat RF on macro (S0 is weak). The project owner later selected this same C0 as thesis-final authority. That decision is in `artifacts/prediction/final/FINALIZATION_DECISION.json`.

## Limitations (kept explicit)

- **UCI S0** has no G1/G2 and underperforms RF. It is a real limitation, not a win.
- **OULAD 100%** history length is associated with Withdrawn. Do not treat 100% as a clean academic-risk endpoint.

## Code and reports

| Component | Path |
|---|---|
| Model / adapters / inference | `src/prediction/` |
| Config / registry | `configs/prediction/hybrid_final.json`, `registry.json` |
| Canonical report | `reports/prediction/final/FINAL_PREDICTION_MODEL_REPORT.md` |
| Result tables | `reports/prediction/final/uci_final.csv`, `oulad_final.csv` |
| Decision + audits | `artifacts/prediction/final/` |

Raw UCI/OULAD files are not bundled. Active source contains the feature-building pipeline (`src/prediction/data/oulad_features.py`): cutoff-safe weekly tensors and D3 aggregates from local raw tables. Scalers remain FIT-only and are not fit on VALID/test.

## Checks (no training, no HPO, no outer)

```powershell
python project.py prediction status
python project.py prediction registry
python project.py prediction validate
pytest tests/prediction -q
```

## Recommendation and database

`src/recommend_hybrid/` remains downstream and is unchanged except for the public `PredictionResult` contract. Database artifacts are historical provenance, not the prediction authority.
