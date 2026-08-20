# PROJECT — Hybrid CNN–BiLSTM (current authority)

This is the current project map for the thesis prediction system.
Historical multiclass / `cnn_bilstm_*` / H1 descriptions live only under
`reports/prediction/historical/pre_phase4/PROJECT_LEGACY.md`.

## Task

Binary academic-risk prediction.

| Dataset | Target | Information states of one fitted model |
|---|---|---|
| UCI | `G3 < 10` | S0 → S1 → S2 |
| OULAD | Fail / Withdrawn | 20% → 35% → 50% → 75% → 100% |

## Model

- Public name: **Hybrid** (`model_id = hybrid`, class `Hybrid`)
- Architecture: **Hybrid CNN–BiLSTM** — parallel CNN ∥ BiLSTM, corrected availability, 3-way masked softmax, one binary logit
- Shared widths: `d_fuse=128`, `cnn_channels=64`, `bilstm_hidden=128`
- One architecture for UCI and OULAD. No dataset-specific topology. No separate OULAD-100 model.

## Active baselines

`LR / DT / RF / SVM / MLP`

XGBoost is historical only, not active.

## Evaluation

- Thesis-final numbers: Phase 4 robust inner 3×3
- Outer was **not** opened
- Research gate recorded `NOT_READY_FOR_FINAL_EVAL` (UCI S0 vs RF). The project owner later selected this Hybrid CNN–BiLSTM as thesis-final authority.

## Known limitations

- **UCI S0** has no G1/G2 and underperforms RF.
- **OULAD 100%** history length is associated with Withdrawn.

## Where to read

| Item | Path |
|---|---|
| Canonical report | `reports/prediction/final/FINAL_PREDICTION_MODEL_REPORT.md` |
| Tables | `reports/prediction/final/uci_final.csv`, `oulad_final.csv` |
| Decision | `artifacts/prediction/final/FINALIZATION_DECISION.json` |
| Config | `configs/prediction/hybrid_final.json` |
| Code | `src/prediction/` |
| Registry of current vs historical | `reports/CURRENT_REPORTS.md` |
