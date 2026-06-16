# Technical Experiment Summary

- Mode: full configured run
- Student datasets only: student-mat and/or student-por; student-combine is not used.
- ADASYN is not used. Mixed categorical/numerical oversampling uses SMOTENC; random oversampling duplicates rows.
- Locked test is evaluated after CV/OOF threshold tuning and is not used for model selection.
- Required metrics are exported: Accuracy, Macro Precision, Macro Recall, Macro F1, Recall Low, F1 Low, RMSE, R2.
- Main RMSE/R2 use a fixed class-to-point mapping; deep regression-head RMSE/R2 are exported separately when available.

## Baseline CV Top Rows

| dataset | scenario | strategy | model | macro_f1 | recall_low | f1_low | rmse | r2 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| student-mat | late | smotenc | xgboost | 0.8960 | 0.9135 | 0.8752 | 2.2241 | 0.7524 |
| student-mat | late | none | xgboost | 0.8871 | 0.8462 | 0.8541 | 2.3272 | 0.7292 |
| student-mat | late | class_weight | xgboost | 0.8848 | 0.9231 | 0.8764 | 2.2889 | 0.7381 |
| student-mat | late | random_oversampling | xgboost | 0.8838 | 0.9327 | 0.8817 | 2.2745 | 0.7410 |
| student-mat | late | class_weight | random_forest | 0.8785 | 0.8846 | 0.8504 | 2.4086 | 0.7089 |
| student-por | late | random_oversampling | random_forest | 0.8675 | 0.7625 | 0.7737 | 2.1142 | 0.6701 |
| student-por | late | none | xgboost | 0.8641 | 0.6500 | 0.7551 | 2.0600 | 0.6883 |
| student-por | late | none | random_forest | 0.8639 | 0.7000 | 0.7580 | 2.1076 | 0.6745 |
| student-por | late | random_oversampling | xgboost | 0.8618 | 0.8750 | 0.7636 | 2.2566 | 0.6261 |
| student-por | late | class_weight | xgboost | 0.8617 | 0.8750 | 0.7636 | 2.2566 | 0.6261 |
| student-mat | late | smotenc | random_forest | 0.8596 | 0.8750 | 0.8322 | 2.5739 | 0.6683 |
| student-mat | late | random_oversampling | random_forest | 0.8581 | 0.8558 | 0.8325 | 2.5521 | 0.6713 |

## CNN-BiLSTM CV Top Rows

| dataset | scenario | strategy | model | macro_f1 | recall_low | f1_low | rmse | r2 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| student-por | late | none | cnn_bilstm_v27 | 0.3192 | 0.1250 | 0.1754 | 3.2191 | -0.0323 |
| student-mat | midterm | focal_loss | cnn_bilstm_v27 | 0.2302 | 0.5000 | 0.2476 | 6.0747 | -0.7870 |
| student-mat | late | smotenc_focal_loss | cnn_bilstm_v27 | 0.2189 | 0.5000 | 0.2476 | 6.3433 | -1.1104 |
| student-por | midterm | smotenc | cnn_bilstm_v27 | 0.2168 | 0.8125 | 0.4585 | 6.6296 | -3.7722 |
| student-por | late | smotenc_focal_loss | cnn_bilstm_v27 | 0.2161 | 0.9875 | 0.3579 | 6.5607 | -3.7076 |
| student-por | midterm | random_oversampling | cnn_bilstm_v27 | 0.2072 | 0.7500 | 0.4368 | 6.7034 | -3.8433 |
| student-mat | early | focal_loss | cnn_bilstm_v27 | 0.2025 | 0.5000 | 0.2476 | 6.1136 | -0.8072 |
| student-mat | early | class_weight | cnn_bilstm_v27 | 0.1918 | 0.5000 | 0.2476 | 5.9687 | -0.7338 |
| student-mat | early | random_oversampling | cnn_bilstm_v27 | 0.1918 | 0.5000 | 0.2476 | 5.9687 | -0.7338 |
| student-mat | early | none | cnn_bilstm_v27 | 0.1918 | 0.5000 | 0.2476 | 5.9687 | -0.7338 |
| student-mat | late | focal_loss | cnn_bilstm_v27 | 0.1918 | 0.5000 | 0.2476 | 6.1699 | -1.0352 |
| student-mat | early | smotenc | cnn_bilstm_v27 | 0.1918 | 0.5000 | 0.2476 | 5.9687 | -0.7338 |

## Baseline Locked Test Top Rows

| dataset | scenario | strategy | model | macro_f1 | recall_low | f1_low | rmse | r2 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| student-mat | late | none | random_forest | 0.9463 | 0.9231 | 0.9057 | 2.5754 | 0.6873 |
| student-mat | late | smotenc | xgboost | 0.9365 | 0.9615 | 0.8929 | 2.7421 | 0.6455 |
| student-mat | late | random_oversampling | xgboost | 0.9365 | 0.9615 | 0.8929 | 2.7421 | 0.6455 |
| student-mat | late | smotenc | random_forest | 0.9312 | 0.9615 | 0.9091 | 2.6832 | 0.6606 |
| student-mat | late | none | xgboost | 0.9256 | 0.9231 | 0.8727 | 2.7096 | 0.6539 |
| student-mat | late | class_weight | xgboost | 0.9207 | 0.9615 | 0.8929 | 2.7484 | 0.6439 |
| student-mat | late | random_oversampling | random_forest | 0.9202 | 0.9231 | 0.8889 | 2.6499 | 0.6689 |
| student-mat | late | class_weight | random_forest | 0.9197 | 0.8846 | 0.8846 | 2.8435 | 0.6188 |
| student-mat | late | none | logistic_regression | 0.8732 | 0.8846 | 0.8846 | 2.9403 | 0.5924 |
| student-por | late | random_oversampling | xgboost | 0.8586 | 0.9000 | 0.8000 | 2.4444 | 0.4943 |
| student-por | late | class_weight | xgboost | 0.8586 | 0.9000 | 0.8000 | 2.4444 | 0.4943 |
| student-por | late | smotenc | xgboost | 0.8586 | 0.7500 | 0.7895 | 2.1640 | 0.6037 |

## CNN-BiLSTM Locked Test Top Rows

| dataset | scenario | strategy | model | prediction_mode | macro_f1 | recall_low | f1_low | rmse | r2 | regression_head_rmse |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| student-por | midterm | random_oversampling | cnn_bilstm_v27 | low_threshold_tuned | 0.4042 | 0.7500 | 0.6250 | 4.4637 | -0.6863 | 6.4474 |
| student-por | midterm | smotenc_focal_loss | cnn_bilstm_v27 | argmax | 0.3342 | 1.0000 | 0.4444 | 5.4140 | -1.4807 | 6.4554 |
| student-mat | late | none | cnn_bilstm_v27 | argmax | 0.3264 | 0.9231 | 0.5581 | 6.3999 | -0.9310 | 11.2842 |
| student-por | late | smotenc_focal_loss | cnn_bilstm_v27 | argmax | 0.3251 | 0.7500 | 0.5455 | 4.7972 | -0.9477 | 5.7160 |
| student-por | midterm | smotenc | cnn_bilstm_v27 | argmax | 0.3198 | 0.6000 | 0.5714 | 4.9695 | -1.0901 | 6.4380 |
| student-por | midterm | random_oversampling | cnn_bilstm_v27 | argmax | 0.3062 | 0.2500 | 0.3448 | 5.1698 | -1.2620 | 6.4474 |
| student-mat | midterm | none | cnn_bilstm_v27 | argmax | 0.3031 | 0.4615 | 0.4211 | 5.4584 | -0.4046 | 11.1206 |
| student-por | late | class_weight | cnn_bilstm_v27 | argmax | 0.2991 | 0.2000 | 0.1778 | 4.7105 | -0.8779 | 11.2711 |
| student-mat | midterm | smotenc_focal_loss | cnn_bilstm_v27 | argmax | 0.2982 | 0.4231 | 0.5000 | 6.4695 | -0.9732 | 10.8741 |
| student-por | early | random_oversampling | cnn_bilstm_v27 | argmax | 0.2841 | 0.0000 | 0.0000 | 3.7076 | -0.1634 | 6.0343 |
| student-por | late | smotenc | cnn_bilstm_v27 | low_threshold_tuned | 0.2780 | 0.7000 | 0.4000 | 5.4670 | -1.5296 | 5.5402 |
| student-por | late | random_oversampling | cnn_bilstm_v27 | low_threshold_tuned | 0.2752 | 0.6500 | 0.3714 | 5.5335 | -1.5914 | 5.5127 |

## Output Locations

- `reports/final/scenarios/`
- `reports/final/baselines/`
- `reports/final/imbalance/`
- `reports/final/ablation/`