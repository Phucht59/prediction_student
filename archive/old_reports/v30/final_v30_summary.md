# Final V30 Summary

## Protocol

- Scope: only student-mat late and xAPI.
- No ADASYN.
- No student-combine.
- Threshold and temperature choices are tuned from OOF train-pool probabilities only.
- Model selection uses CV/OOF only. Locked test is final evaluation only.
- Regression head is not claimed.
- Runtime config: cv_folds=3, max_epochs=50, patience=10, student_mat_ensemble_seeds=[42, 123, 155, 156, 2025, 7, 99, 200, 300, 500, 1337], xapi_ensemble_seeds=[42, 123, 155, 156, 2025].

## Old/V29/V30 Comparison

| dataset | scenario | source | variant | prediction_mode | macro_f1 | recall_low | f1_low |
| --- | --- | --- | --- | --- | --- | --- | --- |
| student-mat | late | old_champion | sequence_cnn_bilstm_only | low_f1_tuned | 0.9365 | 0.9615 | 0.8929 |
| student-mat | late | v29_selected | seq_kernel1_small | low_f1_tuned | 0.9207 | 0.9615 | 0.8929 |
| student-mat | late | v30_selected | old_seq_default | argmax | 0.9256 | 0.9231 | 0.8727 |
| xapi | xapi | v28_best_deep | gated_fusion_v28 | low_f1_tuned | 0.7541 | 0.8846 | 0.8214 |
| xapi | xapi | v29_selected | xapi_sequence_gated_light | argmax | 0.7507 | 0.8462 | 0.8462 |
| xapi | xapi | v30_selected | xapi_gated_fusion_v30 | balanced_low_macro | 0.7309 | 0.8462 | 0.8148 |

## V30 CV/OOF Selected Models

| dataset | scenario | candidate_id | variant | prediction_mode | temperature | macro_f1_mean | recall_low_mean | f1_low_mean | macro_f1_std |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| student-mat | late | old_seq_default_control | old_seq_default | argmax | 1.0000 | 0.8963 | 0.9232 | 0.8768 | 0.0213 |
| xapi | xapi | xapi_gated_k3_c32_h64 | xapi_gated_fusion_v30 | balanced_low_macro | 1.0000 | 0.7794 | 0.9207 | 0.8692 | 0.0132 |

## V30 Locked-Test Results

| dataset | scenario | candidate_id | variant | prediction_mode | temperature | macro_f1 | recall_low | f1_low | rmse | r2 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| student-mat | late | old_seq_default_control | old_seq_default | argmax | 1.0000 | 0.9256 | 0.9231 | 0.8727 | 2.7096 | 0.6539 |
| xapi | xapi | xapi_gated_k3_c32_h64 | xapi_gated_fusion_v30 | balanced_low_macro | 1.0000 | 0.7309 | 0.8462 | 0.8148 | 3.1853 | 0.5390 |

## Deep vs Baseline Same Scenario

| dataset | scenario | deep_variant | deep_prediction_mode | deep_macro_f1 | deep_recall_low | deep_f1_low | baseline_model | baseline_macro_f1 | macro_f1_gap_deep_minus_baseline |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| student-mat | late | old_seq_default | argmax | 0.9256 | 0.9231 | 0.8727 | xgboost | 0.9469 | -0.0213 |
| xapi | xapi | xapi_gated_fusion_v30 | balanced_low_macro | 0.7309 | 0.8462 | 0.8148 | random_forest | 0.8465 | -0.1157 |

## Conclusion

Final champion is changed only if the CV/OOF-selected V30 row improves the existing best locked-test deep row for the same dataset/scenario.

| dataset | scenario | source | variant | prediction_mode | macro_f1 | recall_low | f1_low |
| --- | --- | --- | --- | --- | --- | --- | --- |
| student-mat | late | old_champion | sequence_cnn_bilstm_only | low_f1_tuned | 0.9365 | 0.9615 | 0.8929 |
| xapi | xapi | v28_best_deep | gated_fusion_v28 | low_f1_tuned | 0.7541 | 0.8846 | 0.8214 |