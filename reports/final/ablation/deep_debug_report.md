# Deep Model Debug Report

- Mode: configured
- Purpose: debug PyTorch deep branch before claiming CNN-BiLSTM as the main model.
- Early scenario has no real sequence; sequence-only rows are skipped.
- Main RMSE/R2 now use class-to-point mapping; regression-head metrics are separate columns.

## Overfit Sanity

| dataset | scenario | variant | status | macro_f1 | recall_low | f1_low | rmse | r2 | regression_head_rmse |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| student-mat | midterm | context_mlp_only | pass | 1.0000 | 1.0000 | 1.0000 | 2.5658 | 0.7029 | 11.9595 |
| student-mat | midterm | sequence_cnn_bilstm_only | fail | 0.8022 | 1.0000 | 0.8649 | 3.4126 | 0.4745 | 12.0915 |
| student-mat | midterm | fusion_cnn_bilstm_context | pass | 1.0000 | 1.0000 | 1.0000 | 2.5658 | 0.7029 | 12.8540 |

## Branch Ablation CV

| dataset | scenario | variant | macro_f1 | recall_low | f1_low | rmse | r2 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| student-mat | midterm | fusion_cnn_bilstm_context | 0.7230 | 0.7212 | 0.7417 | 3.5582 | 0.3888 |
| student-mat | midterm | sequence_cnn_bilstm_only | 0.7229 | 0.6538 | 0.7107 | 3.3921 | 0.4400 |
| student-mat | midterm | context_mlp_only | 0.4398 | 0.3846 | 0.4337 | 5.4108 | -0.4216 |

## Branch Ablation Locked Test

| dataset | scenario | variant | status | macro_f1 | recall_low | f1_low | rmse | r2 | regression_head_rmse |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| student-mat | midterm | fusion_cnn_bilstm_context | evaluated | 0.8035 | 0.7692 | 0.7407 | 3.8570 | 0.2987 | 11.4472 |
| student-mat | midterm | sequence_cnn_bilstm_only | evaluated | 0.7886 | 0.8462 | 0.7586 | 3.6500 | 0.3719 | 11.2287 |
| student-mat | midterm | context_mlp_only | evaluated | 0.4808 | 0.3462 | 0.4091 | 5.2909 | -0.3197 | 11.2757 |