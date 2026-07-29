# OULAD Unified Multi-stage Results

All OULAD model identities use one estimator/checkpoint per outer fold and seed across E1, E2, M1 and L1.

| Stage | Best model | Macro-F1 | CNN-BiLSTM Macro-F1 |
|---|---|---:|---:|
| E1_EARLY_20PCT | xgboost_oulad | 0.7070 | 0.7003 |
| E2_EARLY_35PCT | xgboost_oulad | 0.7524 | 0.7435 |
| M1_MIDDLE_FROZEN | hist_gradient_boosting_oulad | 0.7938 | 0.7852 |
| L1_LATE_75PCT | svm_oulad | 0.8324 | 0.8062 |

M1 retains the historical F2 cutoff definition, but its unified training result is not expected to reproduce the historical frozen F2 score exactly.

Future OULAD is `LOCKED_NOT_EXECUTED`; recommendations remain frozen and no canonical database cutover was performed.
