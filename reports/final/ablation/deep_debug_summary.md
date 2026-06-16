# Deep Model Debug Report

- Mode: configured
- Purpose: debug PyTorch deep branch before claiming CNN-BiLSTM as the main model.
- Early scenario has no real sequence; only context MLP variants are evaluated.
- Thresholds are tuned from OOF train-pool probabilities, never from locked test.
- Baseline-vs-deep rows use CV-selected baselines and same locked test per dataset/scenario.
- Main RMSE/R2 now use class-to-point mapping; regression-head metrics are separate columns.
- Regression head should not be claimed while `regression_head_rmse` remains high.

## Overfit Sanity

| dataset | scenario | variant | config_id | status | macro_f1 | recall_low | f1_low | rmse | r2 | regression_head_rmse |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| student-mat | early | context_mlp_only | default | fail | 0.6470 | 0.6190 | 0.6500 | 5.7189 | -0.3143 | 12.1300 |
| student-mat | early | context_mlp_v2 | dropout0.1_wd0.0001_fs1 | pass | 0.9683 | 1.0000 | 1.0000 | 2.7153 | 0.7037 | 12.1887 |
| student-mat | early | context_mlp_v2 | dropout0.25_wd0.0001_fs1 | pass | 0.9524 | 1.0000 | 1.0000 | 2.8179 | 0.6809 | 12.1655 |
| student-mat | early | context_mlp_v2 | dropout0.25_wd0.001_fs0 | fail | 0.9369 | 0.9524 | 0.9756 | 2.7782 | 0.6898 | 12.1655 |
| student-mat | midterm | sequence_cnn_bilstm_only | default | fail | 0.5334 | 1.0000 | 0.7925 | 4.1385 | 0.3117 | 12.0874 |
| student-mat | midterm | context_mlp_only | default | fail | 0.6470 | 0.6190 | 0.6500 | 5.7189 | -0.3143 | 12.1300 |
| student-mat | midterm | context_mlp_v2 | dropout0.1_wd0.0001_fs1 | pass | 0.9683 | 1.0000 | 1.0000 | 2.7153 | 0.7037 | 12.1887 |
| student-mat | midterm | context_mlp_v2 | dropout0.25_wd0.0001_fs1 | pass | 0.9524 | 1.0000 | 1.0000 | 2.8179 | 0.6809 | 12.1655 |
| student-mat | midterm | context_mlp_v2 | dropout0.25_wd0.001_fs0 | fail | 0.9369 | 0.9524 | 0.9756 | 2.7782 | 0.6898 | 12.1655 |
| student-mat | midterm | fusion_cnn_bilstm_context | default | fail | 0.7505 | 0.5714 | 0.7273 | 3.8032 | 0.4187 | 12.6499 |
| student-mat | late | sequence_cnn_bilstm_only | default | fail | 0.7349 | 1.0000 | 0.9130 | 3.3088 | 0.5600 | 12.0763 |
| student-mat | late | context_mlp_only | default | fail | 0.7601 | 0.7143 | 0.7692 | 3.9716 | 0.3661 | 12.1769 |
| student-mat | late | context_mlp_v2 | dropout0.1_wd0.0001_fs1 | pass | 0.9527 | 0.9524 | 0.9756 | 2.6741 | 0.7126 | 12.1914 |
| student-mat | late | context_mlp_v2 | dropout0.25_wd0.0001_fs1 | pass | 0.9527 | 0.9524 | 0.9756 | 2.6741 | 0.7126 | 12.1862 |
| student-mat | late | context_mlp_v2 | dropout0.25_wd0.001_fs0 | pass | 0.9527 | 0.9524 | 0.9756 | 2.6741 | 0.7126 | 12.1871 |
| student-mat | late | fusion_cnn_bilstm_context | default | fail | 0.8432 | 0.7619 | 0.8421 | 3.2721 | 0.5698 | 12.3506 |
| student-por | early | context_mlp_only | default | fail | 0.7456 | 0.8095 | 0.7907 | 4.3675 | -0.4301 | 12.7486 |
| student-por | early | context_mlp_v2 | dropout0.1_wd0.0001_fs1 | pass | 0.9841 | 1.0000 | 1.0000 | 2.7087 | 0.4499 | 12.7544 |
| student-por | early | context_mlp_v2 | dropout0.25_wd0.0001_fs1 | pass | 0.9527 | 0.9524 | 0.9756 | 2.7646 | 0.4270 | 12.7488 |
| student-por | early | context_mlp_v2 | dropout0.25_wd0.001_fs0 | pass | 0.9527 | 0.9524 | 0.9756 | 2.7646 | 0.4270 | 12.7456 |

## Branch Ablation CV

| dataset | scenario | variant | config_id | macro_f1 | recall_low | f1_low | rmse | r2 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| student-por | late | sequence_cnn_bilstm_only | default | 0.8480 | 0.7875 | 0.7721 | 2.4167 | 0.4160 |
| student-por | late | fusion_cnn_bilstm_context | default | 0.8474 | 0.7875 | 0.7836 | 2.4075 | 0.4196 |
| student-mat | late | sequence_cnn_bilstm_only | default | 0.8338 | 0.8365 | 0.8168 | 3.0669 | 0.5351 |
| student-mat | late | fusion_cnn_bilstm_context | default | 0.8304 | 0.8654 | 0.8127 | 3.1148 | 0.5164 |
| student-mat | midterm | sequence_cnn_bilstm_only | default | 0.7605 | 0.7885 | 0.7647 | 3.3194 | 0.4620 |
| student-mat | midterm | fusion_cnn_bilstm_context | default | 0.7595 | 0.7885 | 0.7573 | 3.4268 | 0.4304 |
| student-por | midterm | fusion_cnn_bilstm_context | default | 0.7584 | 0.6000 | 0.6554 | 2.4914 | 0.3813 |
| student-mat | late | context_mlp_v2 | dropout0.1_wd0.0001_fs1 | 0.7182 | 0.6442 | 0.7118 | 3.5011 | 0.4116 |
| student-por | midterm | sequence_cnn_bilstm_only | default | 0.7163 | 0.4375 | 0.5582 | 2.5086 | 0.3659 |
| student-mat | late | context_mlp_v2 | dropout0.25_wd0.0001_fs1 | 0.6888 | 0.6635 | 0.7142 | 3.7462 | 0.3272 |
| student-por | late | context_mlp_v2 | dropout0.1_wd0.0001_fs1 | 0.6380 | 0.3375 | 0.4118 | 2.8787 | 0.1737 |
| student-por | late | context_mlp_v2 | dropout0.25_wd0.0001_fs1 | 0.5990 | 0.3375 | 0.3785 | 3.1106 | 0.0323 |
| student-por | late | context_mlp_v2 | dropout0.25_wd0.001_fs0 | 0.5743 | 0.4250 | 0.4560 | 3.4632 | -0.2046 |
| student-mat | late | context_mlp_v2 | dropout0.25_wd0.001_fs0 | 0.4860 | 0.4904 | 0.5270 | 4.8467 | -0.1383 |
| student-mat | midterm | context_mlp_v2 | dropout0.25_wd0.0001_fs1 | 0.4692 | 0.4038 | 0.4664 | 5.2382 | -0.3292 |
| student-mat | early | context_mlp_v2 | dropout0.25_wd0.0001_fs1 | 0.4692 | 0.4038 | 0.4664 | 5.2382 | -0.3292 |
| student-por | midterm | context_mlp_v2 | dropout0.1_wd0.0001_fs1 | 0.4557 | 0.3000 | 0.3534 | 3.7881 | -0.4392 |
| student-por | early | context_mlp_v2 | dropout0.1_wd0.0001_fs1 | 0.4557 | 0.3000 | 0.3534 | 3.7881 | -0.4392 |
| student-por | early | context_mlp_v2 | dropout0.25_wd0.0001_fs1 | 0.4536 | 0.3625 | 0.3773 | 3.8462 | -0.4893 |
| student-por | midterm | context_mlp_v2 | dropout0.25_wd0.0001_fs1 | 0.4536 | 0.3625 | 0.3773 | 3.8462 | -0.4893 |

## Low-Class Threshold Tuning

| dataset | scenario | variant | config_id | prediction_mode | threshold_low | macro_f1 | recall_low | f1_low |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| student-mat | early | context_mlp_v2 | dropout0.25_wd0.0001_fs1 | argmax | nan | 0.4700 | 0.4038 | 0.4667 |
| student-mat | early | context_mlp_v2 | dropout0.25_wd0.0001_fs1 | low_f1_tuned | 0.2750 | 0.4537 | 0.6442 | 0.5255 |
| student-mat | early | context_mlp_v2 | dropout0.1_wd0.0001_fs1 | argmax | nan | 0.4438 | 0.3846 | 0.4348 |
| student-mat | early | context_mlp_only | default | low_f1_tuned | 0.3500 | 0.4093 | 0.6154 | 0.5267 |
| student-mat | early | context_mlp_v2 | dropout0.25_wd0.001_fs0 | argmax | nan | 0.4070 | 0.5096 | 0.4796 |
| student-mat | early | context_mlp_only | default | argmax | nan | 0.4060 | 0.4231 | 0.4656 |
| student-mat | early | context_mlp_v2 | dropout0.1_wd0.0001_fs1 | low_f1_tuned | 0.1750 | 0.3906 | 0.7500 | 0.5183 |
| student-mat | early | context_mlp_v2 | dropout0.25_wd0.001_fs0 | low_f1_tuned | 0.1500 | 0.2999 | 0.8846 | 0.5097 |
| student-mat | early | context_mlp_v2 | dropout0.1_wd0.0001_fs1 | low_threshold_tuned | 0.0500 | 0.2545 | 0.9712 | 0.5140 |
| student-mat | early | context_mlp_v2 | dropout0.1_wd0.0001_fs1 | low_recall_priority | 0.0500 | 0.2545 | 0.9712 | 0.5140 |
| student-mat | early | context_mlp_v2 | dropout0.25_wd0.0001_fs1 | low_threshold_tuned | 0.0750 | 0.2359 | 0.9712 | 0.5114 |
| student-mat | early | context_mlp_v2 | dropout0.25_wd0.001_fs0 | low_threshold_tuned | 0.0500 | 0.1796 | 0.9808 | 0.4928 |
| student-mat | early | context_mlp_v2 | dropout0.25_wd0.001_fs0 | low_recall_priority | 0.0500 | 0.1796 | 0.9808 | 0.4928 |
| student-mat | early | context_mlp_v2 | dropout0.25_wd0.0001_fs1 | low_recall_priority | 0.0500 | 0.1693 | 0.9904 | 0.4952 |
| student-mat | early | context_mlp_only | default | low_threshold_tuned | 0.0500 | 0.1651 | 1.0000 | 0.4952 |
| student-mat | early | context_mlp_only | default | low_recall_priority | 0.0500 | 0.1651 | 1.0000 | 0.4952 |
| student-mat | late | fusion_cnn_bilstm_context | default | low_f1_tuned | 0.6000 | 0.8501 | 0.8462 | 0.8381 |
| student-mat | late | sequence_cnn_bilstm_only | default | low_f1_tuned | 0.3000 | 0.8387 | 0.9423 | 0.8412 |
| student-mat | late | sequence_cnn_bilstm_only | default | argmax | nan | 0.8351 | 0.8365 | 0.8208 |
| student-mat | late | fusion_cnn_bilstm_context | default | argmax | nan | 0.8322 | 0.8654 | 0.8145 |

## Branch Ablation Locked Test

| dataset | scenario | variant | config_id | prediction_mode | status | macro_f1 | recall_low | f1_low | rmse | r2 | regression_head_rmse |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| student-mat | late | sequence_cnn_bilstm_only | default | low_f1_tuned | evaluated | 0.9365 | 0.9615 | 0.8929 | 2.7421 | 0.6455 | 11.1351 |
| student-mat | late | sequence_cnn_bilstm_only | default | argmax | evaluated | 0.9034 | 0.8462 | 0.8302 | 2.9296 | 0.5954 | 11.1351 |
| student-por | late | sequence_cnn_bilstm_only | default | low_f1_tuned | evaluated | 0.8783 | 0.9000 | 0.8182 | 2.4087 | 0.5090 | 11.9944 |
| student-mat | late | sequence_cnn_bilstm_only | default | low_threshold_tuned | evaluated | 0.8750 | 1.0000 | 0.8125 | 3.2994 | 0.4868 | 11.1351 |
| student-por | late | sequence_cnn_bilstm_only | default | argmax | evaluated | 0.8696 | 0.7500 | 0.7895 | 2.1737 | 0.6001 | 11.9944 |
| student-mat | late | fusion_cnn_bilstm_context | default | argmax | evaluated | 0.8689 | 0.9231 | 0.8571 | 2.8245 | 0.6239 | 11.9174 |
| student-mat | late | fusion_cnn_bilstm_context | default | low_f1_tuned | evaluated | 0.8687 | 0.8846 | 0.8519 | 2.9772 | 0.5821 | 11.9174 |
| student-mat | late | sequence_cnn_bilstm_only | default | low_recall_priority | evaluated | 0.8540 | 1.0000 | 0.7879 | 3.4823 | 0.4283 | 11.1351 |
| student-por | late | fusion_cnn_bilstm_context | default | low_f1_tuned | evaluated | 0.8490 | 0.6500 | 0.7429 | 2.0829 | 0.6328 | 12.8124 |
| student-por | midterm | sequence_cnn_bilstm_only | default | argmax | evaluated | 0.8228 | 0.6500 | 0.7429 | 2.3806 | 0.5204 | 12.0709 |
| student-por | late | fusion_cnn_bilstm_context | default | argmax | evaluated | 0.8200 | 0.5500 | 0.6667 | 2.2566 | 0.5690 | 12.8124 |
| student-por | late | fusion_cnn_bilstm_context | default | low_recall_priority | evaluated | 0.8133 | 0.8000 | 0.6809 | 2.5690 | 0.4414 | 12.8124 |
| student-por | late | fusion_cnn_bilstm_context | default | low_threshold_tuned | evaluated | 0.8133 | 0.8000 | 0.6809 | 2.5690 | 0.4414 | 12.8124 |
| student-por | late | sequence_cnn_bilstm_only | default | low_recall_priority | evaluated | 0.7990 | 1.0000 | 0.6667 | 3.0763 | 0.1991 | 11.9944 |
| student-por | late | sequence_cnn_bilstm_only | default | low_threshold_tuned | evaluated | 0.7990 | 1.0000 | 0.6667 | 3.0763 | 0.1991 | 11.9944 |
| student-mat | late | context_mlp_v2 | dropout0.25_wd0.0001_fs1 | argmax | evaluated | 0.7886 | 0.8077 | 0.7925 | 3.5113 | 0.4188 | 11.2782 |
| student-mat | midterm | sequence_cnn_bilstm_only | default | argmax | evaluated | 0.7886 | 0.8462 | 0.7586 | 3.6500 | 0.3719 | 11.2195 |
| student-mat | late | fusion_cnn_bilstm_context | default | low_recall_priority | evaluated | 0.7802 | 1.0000 | 0.7647 | 3.5725 | 0.3983 | 11.9174 |
| student-mat | late | fusion_cnn_bilstm_context | default | low_threshold_tuned | evaluated | 0.7802 | 1.0000 | 0.7647 | 3.5725 | 0.3983 | 11.9174 |
| student-por | midterm | sequence_cnn_bilstm_only | default | low_f1_tuned | evaluated | 0.7799 | 0.8000 | 0.6667 | 2.7383 | 0.3654 | 12.0709 |

## Baseline Vs Deep

| dataset | scenario | baseline_model | baseline_strategy | baseline_locked_macro_f1 | deep_variant | deep_config_id | deep_prediction_mode | deep_locked_macro_f1 | macro_f1_gap_deep_minus_baseline | recall_low_gap_deep_minus_baseline |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| student-por | early | xgboost | class_weight | 0.3823 | context_mlp_v2 | dropout0.25_wd0.001_fs0 | low_f1_tuned | 0.4758 | 0.0935 | -0.1000 |
| student-por | early | xgboost | class_weight | 0.3823 | context_mlp_v2 | dropout0.25_wd0.001_fs0 | argmax | 0.4687 | 0.0864 | -0.1500 |
| student-por | late | random_forest | random_oversampling | 0.8379 | sequence_cnn_bilstm_only | default | low_f1_tuned | 0.8783 | 0.0404 | 0.2500 |
| student-por | late | random_forest | random_oversampling | 0.8379 | sequence_cnn_bilstm_only | default | argmax | 0.8696 | 0.0317 | 0.1000 |
| student-por | early | xgboost | class_weight | 0.3823 | context_mlp_only | default | low_f1_tuned | 0.4139 | 0.0316 | 0.1000 |
| student-por | early | xgboost | class_weight | 0.3823 | context_mlp_v2 | dropout0.25_wd0.0001_fs1 | argmax | 0.4066 | 0.0242 | -0.2000 |
| student-mat | early | random_forest | random_oversampling | 0.4791 | context_mlp_v2 | dropout0.25_wd0.0001_fs1 | argmax | 0.4994 | 0.0203 | -0.1538 |
| student-por | late | random_forest | random_oversampling | 0.8379 | fusion_cnn_bilstm_context | default | low_f1_tuned | 0.8490 | 0.0110 | 0.0000 |
| student-mat | early | random_forest | random_oversampling | 0.4791 | context_mlp_v2 | dropout0.1_wd0.0001_fs1 | argmax | 0.4835 | 0.0044 | -0.1538 |
| student-por | early | xgboost | class_weight | 0.3823 | context_mlp_v2 | dropout0.25_wd0.001_fs0 | low_threshold_tuned | 0.3843 | 0.0020 | 0.2500 |
| student-por | early | xgboost | class_weight | 0.3823 | context_mlp_v2 | dropout0.25_wd0.001_fs0 | low_recall_priority | 0.3843 | 0.0020 | 0.2500 |
| student-mat | late | xgboost | smotenc | 0.9365 | sequence_cnn_bilstm_only | default | low_f1_tuned | 0.9365 | 0.0000 | 0.0000 |
| student-por | midterm | xgboost | smotenc | 0.8228 | sequence_cnn_bilstm_only | default | argmax | 0.8228 | 0.0000 | 0.0000 |
| student-por | early | xgboost | class_weight | 0.3823 | context_mlp_v2 | dropout0.1_wd0.0001_fs1 | low_f1_tuned | 0.3684 | -0.0140 | 0.1500 |
| student-por | early | xgboost | class_weight | 0.3823 | context_mlp_v2 | dropout0.1_wd0.0001_fs1 | argmax | 0.3682 | -0.0142 | -0.2000 |
| student-mat | early | random_forest | random_oversampling | 0.4791 | context_mlp_v2 | dropout0.25_wd0.0001_fs1 | low_f1_tuned | 0.4638 | -0.0154 | 0.0385 |
| student-por | late | random_forest | random_oversampling | 0.8379 | fusion_cnn_bilstm_context | default | argmax | 0.8200 | -0.0179 | -0.1000 |
| student-por | early | xgboost | class_weight | 0.3823 | context_mlp_v2 | dropout0.25_wd0.0001_fs1 | low_f1_tuned | 0.3628 | -0.0196 | 0.1500 |
| student-por | late | random_forest | random_oversampling | 0.8379 | fusion_cnn_bilstm_context | default | low_threshold_tuned | 0.8133 | -0.0246 | 0.1500 |
| student-por | late | random_forest | random_oversampling | 0.8379 | fusion_cnn_bilstm_context | default | low_recall_priority | 0.8133 | -0.0246 | 0.1500 |