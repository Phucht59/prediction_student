# Strategy B Phase E-Prediction conclusion

## Development-only stability results

| candidate_id | oof_macro_f1 | accuracy | macro_precision | macro_recall | high_f1 | macro_pr_auc | rmse | r2 |
|---|---|---|---|---|---|---|---|---|
| M1 | 0.899955 | 0.892371 | 0.907908 | 0.892426 | 0.933247 | 0.952640 | 2.460936 | 0.706501 |
| R0 | 0.898836 | 0.892411 | 0.907759 | 0.893484 | 0.924551 | 0.846130 | 2.008550 | 0.804957 |
| M2 | 0.890102 | 0.882887 | 0.903457 | 0.879777 | 0.924551 | 0.960194 | 2.360472 | 0.730479 |
| N0 | 0.850365 | 0.846190 | 0.860605 | 0.853494 | 0.869358 | 0.950966 | 2.463168 | 0.706730 |
| N1 | 0.838316 | 0.831518 | 0.843495 | 0.862127 | 0.870142 | 0.945725 | 2.432924 | 0.712777 |

- `final_overall_model`: **R0**.
- `final_thesis_hybrid_model`: **N0**.
- Final model family and configuration were selected and frozen using nested development evidence. No untouched external confirmation dataset was available.
- No legacy-observed-79 records, external labels, recommendation Phase D, or conditional branch was used.
- Strict validation: **PASS**.
