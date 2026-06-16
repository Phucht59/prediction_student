# Final Reports

This folder keeps the current final decision for the xAPI deep model.

## xAPI Final Champion

| Dataset | Model | Prediction mode | Macro F1 | Recall Low | F1 Low |
|---|---|---|---:|---:|---:|
| xAPI | gated_fusion_v28 | low_f1_tuned | 0.7541 | 0.8846 | 0.8214 |

## Protocol

- The final xAPI model is CNN-BiLSTM with gated context fusion.
- Locked test is used only for final evaluation.
- Threshold tuning uses CV/OOF probabilities, not locked test.
- Machine-learning baselines are comparison-only.
- No teacher model, distillation, pseudo-labeling, baseline probabilities, or feature-importance outputs are used for deep training.
- ADASYN is not used.
- student-combine is not used.
- Regression head is not claimed.

Primary files:
- `final_model_manifest.json`
- `final_deep_results_table.csv`
- `final_baseline_comparison.csv`
- `final_prediction_model_report.md`
- `final_thesis_ready_summary.md`
- `xapi_3class_final_report.txt`
