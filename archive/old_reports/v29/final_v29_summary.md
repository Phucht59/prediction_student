# Final V29 Summary

## Protocol

- Controlled V29 ablation: old sequence branch, small kernel/loss changes, and light xAPI fusion only.
- No ADASYN.
- No student-combine.
- Thresholds are tuned from OOF train-pool probabilities, never locked test.
- Model selection uses CV/OOF only: Macro F1 mean, then Recall Low, F1 Low, then lower fold std.
- Locked test is final evaluation only for the CV/OOF-selected row per dataset/scenario.
- Regression head is not claimed; RMSE/R2 are mapped-class diagnostics only.
- Runtime config: cv_folds=5, max_epochs=60, patience=12, ensemble_seeds=[42, 123, 155, 156, 2025].

## Old Champion vs V28 vs V29

| dataset | scenario | source | variant | prediction_mode | macro_f1 | recall_low | f1_low |
| --- | --- | --- | --- | --- | --- | --- | --- |
| student-mat | late | old_champion | sequence_cnn_bilstm_only | low_f1_tuned | 0.9365 | 0.9615 | 0.8929 |
| student-mat | late | v28_selected | sequence_cnn_bilstm_v28_focal | low_f1_tuned | 0.8797 | 0.9231 | 0.8727 |
| student-mat | late | v29_selected | seq_kernel1_small | low_f1_tuned | 0.9207 | 0.9615 | 0.8929 |
| student-por | late | old_champion | sequence_cnn_bilstm_only | low_f1_tuned | 0.8783 | 0.9000 | 0.8182 |
| student-por | late | v28_selected | sequence_cnn_bilstm_v28_focal | low_f1_tuned | 0.8595 | 0.6000 | 0.7500 |
| student-por | late | v29_selected | old_seq_default_ensemble | low_f1_tuned | 0.8779 | 0.8500 | 0.8095 |
| student-por | midterm | old_champion | sequence_cnn_bilstm_only | argmax | 0.8228 | 0.6500 | 0.7429 |
| student-por | midterm | v28_selected | sequence_cnn_bilstm_v28_focal | low_f1_tuned | 0.8228 | 0.6500 | 0.7429 |
| student-por | midterm | v29_selected | old_seq_default | balanced_low_macro | 0.7799 | 0.8000 | 0.6667 |
| xapi | xapi | v28_selected | gated_fusion_v28 | low_f1_tuned | 0.7541 | 0.8846 | 0.8214 |
| xapi | xapi | v29_selected | xapi_sequence_gated_light | argmax | 0.7507 | 0.8462 | 0.8462 |

## V29 CV/OOF Selected Models

| dataset | scenario | candidate_id | variant | prediction_mode | macro_f1_mean | recall_low_mean | f1_low_mean | macro_f1_std |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| student-mat | late | seq_kernel1_small | seq_kernel1_small | low_f1_tuned | 0.8931 | 0.9324 | 0.8867 | 0.0291 |
| student-por | late | old_seq_default_ensemble | old_seq_default_ensemble | low_f1_tuned | 0.8905 | 0.8250 | 0.8068 | 0.0426 |
| student-por | midterm | old_seq_default | old_seq_default | balanced_low_macro | 0.7957 | 0.8000 | 0.7210 | 0.0201 |
| xapi | xapi | xapi_gated_light | xapi_sequence_gated_light | argmax | 0.7924 | 0.9110 | 0.8729 | 0.0297 |

## V29 Locked-Test Results

| dataset | scenario | candidate_id | variant | prediction_mode | macro_f1 | recall_low | f1_low | rmse | r2 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| student-mat | late | seq_kernel1_small | seq_kernel1_small | low_f1_tuned | 0.9207 | 0.9615 | 0.8929 | 2.7484 | 0.6439 |
| student-por | late | old_seq_default_ensemble | old_seq_default_ensemble | low_f1_tuned | 0.8779 | 0.8500 | 0.8095 | 2.2992 | 0.5526 |
| student-por | midterm | old_seq_default | old_seq_default | balanced_low_macro | 0.7799 | 0.8000 | 0.6667 | 2.7383 | 0.3654 |
| xapi | xapi | xapi_gated_light | xapi_sequence_gated_light | argmax | 0.7507 | 0.8462 | 0.8462 | 3.0208 | 0.5854 |

## V29 Ensemble OOF Results

| dataset | scenario | candidate_id | prediction_mode | macro_f1_mean | recall_low_mean | f1_low_mean | macro_f1_std |
| --- | --- | --- | --- | --- | --- | --- | --- |
| student-mat | late | old_seq_default_ensemble | argmax | 0.8816 | 0.8943 | 0.8651 | 0.0247 |
| student-mat | late | old_seq_default_ensemble | low_f1_tuned | 0.8842 | 0.8943 | 0.8693 | 0.0288 |
| student-mat | late | old_seq_default_ensemble | low_recall_priority | 0.8333 | 1.0000 | 0.8141 | 0.0411 |
| student-mat | late | old_seq_default_ensemble | balanced_low_macro | 0.8819 | 0.9229 | 0.8687 | 0.0292 |
| student-por | late | old_seq_default_ensemble | argmax | 0.8806 | 0.7625 | 0.7808 | 0.0341 |
| student-por | late | old_seq_default_ensemble | low_f1_tuned | 0.8905 | 0.8250 | 0.8068 | 0.0426 |
| student-por | late | old_seq_default_ensemble | low_recall_priority | 0.7799 | 0.9750 | 0.6076 | 0.0238 |
| student-por | late | old_seq_default_ensemble | balanced_low_macro | 0.8905 | 0.8250 | 0.8068 | 0.0426 |
| student-por | midterm | old_seq_default_ensemble | argmax | 0.7859 | 0.7125 | 0.7221 | 0.0366 |
| student-por | midterm | old_seq_default_ensemble | low_f1_tuned | 0.7859 | 0.7125 | 0.7221 | 0.0366 |
| student-por | midterm | old_seq_default_ensemble | low_recall_priority | 0.6768 | 0.9875 | 0.5738 | 0.0465 |
| student-por | midterm | old_seq_default_ensemble | balanced_low_macro | 0.7603 | 0.9000 | 0.6906 | 0.0227 |
| xapi | xapi | old_seq_default_ensemble | argmax | 0.6666 | 0.8519 | 0.7888 | 0.0308 |
| xapi | xapi | old_seq_default_ensemble | low_f1_tuned | 0.6552 | 0.9410 | 0.7925 | 0.0234 |
| xapi | xapi | old_seq_default_ensemble | low_recall_priority | 0.6552 | 0.9410 | 0.7925 | 0.0234 |
| xapi | xapi | old_seq_default_ensemble | balanced_low_macro | 0.6552 | 0.9410 | 0.7925 | 0.0234 |

## Deep vs Baseline Same Scenario

| dataset | scenario | deep_variant | deep_prediction_mode | deep_macro_f1 | deep_recall_low | deep_f1_low | baseline_model | baseline_macro_f1 | macro_f1_gap_deep_minus_baseline |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| student-mat | late | seq_kernel1_small | low_f1_tuned | 0.9207 | 0.9615 | 0.8929 | xgboost | 0.9469 | -0.0262 |
| student-por | late | old_seq_default_ensemble | low_f1_tuned | 0.8779 | 0.8500 | 0.8095 | xgboost | 0.8411 | 0.0368 |
| student-por | midterm | old_seq_default | balanced_low_macro | 0.7799 | 0.8000 | 0.6667 | xgboost | 0.7659 | 0.0140 |
| xapi | xapi | xapi_sequence_gated_light | argmax | 0.7507 | 0.8462 | 0.8462 | random_forest | 0.8465 | -0.0959 |

## Conclusion

Use the best controlled deep row per dataset/scenario from the comparison table for thesis reporting. Do not replace a stronger old champion with a weaker V29 row; V29 is accepted only where its CV/OOF-selected locked result is competitive or better.

| dataset | scenario | source | variant | prediction_mode | macro_f1 | recall_low | f1_low |
| --- | --- | --- | --- | --- | --- | --- | --- |
| student-mat | late | old_champion | sequence_cnn_bilstm_only | low_f1_tuned | 0.9365 | 0.9615 | 0.8929 |
| student-por | late | old_champion | sequence_cnn_bilstm_only | low_f1_tuned | 0.8783 | 0.9000 | 0.8182 |
| student-por | midterm | old_champion | sequence_cnn_bilstm_only | argmax | 0.8228 | 0.6500 | 0.7429 |
| xapi | xapi | v28_selected | gated_fusion_v28 | low_f1_tuned | 0.7541 | 0.8846 | 0.8214 |