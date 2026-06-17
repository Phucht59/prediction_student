# Final Project Status

## Prediction Model

Prediction model selection is finalized and unchanged.

| Dataset | Scenario | Final model | Prediction mode | Macro F1 | Recall Low | F1 Low |
|---|---|---|---|---:|---:|---:|
| student-mat | late | sequence_cnn_bilstm_only | low_f1_tuned | 0.9365 | 0.9615 | 0.8929 |
| student-por | late | sequence_cnn_bilstm_only | low_f1_tuned | 0.8783 | 0.9000 | 0.8182 |
| student-por | midterm | sequence_cnn_bilstm_only | argmax | 0.8228 | 0.6500 | 0.7429 |
| xAPI | default | gated_fusion_v28 | low_f1_tuned | 0.7541 | 0.8846 | 0.8214 |

## Recommender

RA-HLPR is finalized for xAPI and student-por. It remains downstream of CNN-BiLSTM prediction and is not collaborative filtering.

Student-Mat recommender is pending because the final prediction checkpoint metadata is missing:

`models/saved/final/student-mat_3class_ensemble_features.json`

## Final Reports

- `reports/final/final_model_manifest.json`
- `reports/final/final_deep_results_table.csv`
- `reports/final/final_baseline_comparison.csv`
- `reports/final/final_prediction_model_report.md`
- `reports/final/final_thesis_ready_summary.md`
- `reports/final/final_recommender_report.md`
- `reports/final/final_recommender_thesis_summary_vi.md`
- `reports/final/recommender_model_design.md`
- `reports/final/LUAN_VAN_HOAN_CHINH_FINAL.docx`

## Known Limitations

- xAPI deep model remains below the Random Forest baseline in Macro F1, while keeping high Recall Low.
- Recommender evaluation is offline and based on weak-supervision/rule-based references.
- There is no real post-recommendation student feedback dataset, so causal improvement is not claimed.
- Regression head is not claimed as a final result.

## Next Step

Use these artifacts in the thesis report. Do not run additional V33/V34-style experiments unless a new requirement comes from the advisor.
