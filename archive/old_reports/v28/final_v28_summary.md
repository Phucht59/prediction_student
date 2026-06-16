# Final V28 Summary

## Protocol

- No ADASYN.
- No student-combine.
- Thresholds are tuned from OOF train-pool probabilities, never locked test.
- Model selection uses CV/OOF only. Locked test is final evaluation only.
- Regression head is not claimed because prior regression-head RMSE remained high; V28 reports classification-first results.
- Runtime config: cv_folds=2, max_epochs=20, ensemble_seeds=[42, 123, 155, 156, 2025].
- Baseline package path used: xgboost.

## Old Deep Champions From deep_debug_summary

| dataset | scenario | variant | prediction_mode | macro_f1 | recall_low | f1_low |
| --- | --- | --- | --- | --- | --- | --- |
| student-mat | late | sequence_cnn_bilstm_only | low_f1_tuned | 0.9365 | 0.9615 | 0.8929 |
| student-por | late | sequence_cnn_bilstm_only | low_f1_tuned | 0.8783 | 0.9000 | 0.8182 |
| student-por | midterm | sequence_cnn_bilstm_only | argmax | 0.8228 | 0.6500 | 0.7429 |
| student-mat | midterm | sequence_cnn_bilstm_only | argmax | 0.7886 | 0.8462 | 0.7586 |
| student-mat | early | context_mlp_v2 | argmax | 0.4994 | 0.3077 | 0.3810 |
| student-por | early | context_mlp_v2 | low_f1_tuned | 0.4758 | 0.3000 | 0.3429 |

## New V28 Locked-Test Results

| dataset | scenario | variant | candidate_id | prediction_mode | macro_f1 | recall_low | f1_low |
| --- | --- | --- | --- | --- | --- | --- | --- |
| student-mat | late | sequence_cnn_bilstm_v28_focal | seq_k2_c64_h96_attn_cbf | low_f1_tuned | 0.8797 | 0.9231 | 0.8727 |
| student-por | late | sequence_cnn_bilstm_v28_focal | seq_k2_c64_h96_attn_cbf | low_f1_tuned | 0.8595 | 0.6000 | 0.7500 |
| student-por | midterm | sequence_cnn_bilstm_v28_focal | seq_k2_c64_h96_attn_cbf | low_f1_tuned | 0.8228 | 0.6500 | 0.7429 |
| xapi | xapi | gated_fusion_v28 | gated_k3_c32_h64_attn_cw | low_f1_tuned | 0.7541 | 0.8846 | 0.8214 |

## Seed Ensemble Locked-Test Check

| dataset | scenario | variant | candidate_id | prediction_mode | macro_f1 | recall_low | f1_low |
| --- | --- | --- | --- | --- | --- | --- | --- |
| student-mat | late | sequence_cnn_bilstm_v28_focal | seq_k2_c64_h96_attn_cbf | low_f1_tuned | 0.8797 | 0.9231 | 0.8727 |
| student-por | late | sequence_cnn_bilstm_v28_focal | seq_k2_c64_h96_attn_cbf | low_f1_tuned | 0.8609 | 0.6500 | 0.7879 |
| student-por | midterm | sequence_cnn_bilstm_v28_focal | seq_k2_c64_h96_attn_cbf | low_f1_tuned | 0.8228 | 0.6500 | 0.7429 |

## Deep Vs Baseline Same Scenario

| dataset | scenario | deep_variant | deep_result_type | deep_prediction_mode | macro_f1 | recall_low | f1_low | baseline_model | baseline_macro_f1 | macro_f1_gap_deep_minus_baseline |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| student-mat | late | sequence_cnn_bilstm_v28_focal | selected_seed | low_f1_tuned | 0.8797 | 0.9231 | 0.8727 | xgboost | 0.9469 | -0.0672 |
| student-por | late | sequence_cnn_bilstm_v28_focal | selected_seed | low_f1_tuned | 0.8595 | 0.6000 | 0.7500 | xgboost | 0.8411 | 0.0184 |
| student-por | midterm | sequence_cnn_bilstm_v28_focal | selected_seed | low_f1_tuned | 0.8228 | 0.6500 | 0.7429 | xgboost | 0.7659 | 0.0569 |
| xapi | xapi | gated_fusion_v28 | selected_seed | low_f1_tuned | 0.7541 | 0.8846 | 0.8214 | random_forest | 0.8465 | -0.0924 |

## Conclusion

Selected deep architecture for thesis reporting by CV/OOF: `sequence_cnn_bilstm_v28_focal` from `student-mat/late` with CV/OOF Macro F1=0.8804, Recall Low=0.9327, F1 Low=0.8778. For each dataset/scenario, use the corresponding CV-selected row in `deep_v28_selection.csv`; do not substitute models based on locked-test ranking.