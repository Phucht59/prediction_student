# UCI regression metrics

Classification is the primary thesis task. RMSE and R² are secondary
continuous-G3 diagnostics and are reported only where the registered
regression head produced a scientifically valid aggregate.

| Dataset | Candidate | Status | RMSE | R² | Scope |
|---|---|---|---:|---:|---|
| student_mat | cnn_bilstm_v5_1_transfer_selected_ensemble | NOT_POOLABLE |  |  | outer folds selected zero-weight regression in part of the ensemble; reporting an official pooled value would mix trained and untrained heads |
| student_mat | bilstm_only_v5_1 | SEED_LEVEL_DIAGNOSTIC_NOT_OFFICIAL_POOLED | 3.8222 | 0.3005 | diagnostic only; not substituted for the official classification result |
| student_mat | cnn_bilstm_v5_1 | SEED_LEVEL_DIAGNOSTIC_NOT_OFFICIAL_POOLED | 3.6049 | 0.3779 | diagnostic only; not substituted for the official classification result |
| student_mat | cnn_only_v5_1 | SEED_LEVEL_DIAGNOSTIC_NOT_OFFICIAL_POOLED | 3.9850 | 0.2112 | diagnostic only; not substituted for the official classification result |
| student_por | cnn_bilstm_v5_1_ensemble | OFFICIAL_POOLED_OOF | 2.3497 | 0.4702 | secondary G3 regression metric; classification remains primary |
| student_por | bilstm_only_v5_1 | SEED_LEVEL_DIAGNOSTIC_NOT_OFFICIAL_POOLED | 2.5949 | 0.3458 | diagnostic only; not substituted for the official classification result |
| student_por | cnn_bilstm_v5_1 | SEED_LEVEL_DIAGNOSTIC_NOT_OFFICIAL_POOLED | 2.7282 | 0.2772 | diagnostic only; not substituted for the official classification result |
| student_por | cnn_only_v5_1 | SEED_LEVEL_DIAGNOSTIC_NOT_OFFICIAL_POOLED | 3.1541 | 0.0303 | diagnostic only; not substituted for the official classification result |
