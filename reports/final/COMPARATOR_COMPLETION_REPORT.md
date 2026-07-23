# Comparator Completion Report

Protocol: `final-comparator-completion-20260723-v1`.

Technical amendments 001–002 align ECE with the frozen dataset-specific contracts: 15-bin multiclass confidence for UCI and 10-bin At-risk probability for OULAD. They change no model selection or probability.

The initial release contained 252 serialized missing fields. Frozen probabilities were used for DERIVE_ONLY rows; no complete replay bundle qualified for REPLAY_INFERENCE. Student-Mat/Student-Por XGBoost and all six unified OULAD ML comparators were trained as registered completion models.

Historical OULAD LR/HGB/XGBoost summaries were DO_NOT_IMPORT because record-level native probabilities and a complete unified-contract replay bundle were absent.

## Training

| Dataset | Model | Action | Outer folds | Seeds | Status | Runtime (s) |
|---|---|---|---:|---:|---|---:|
| student_mat | XGBoost | TRAIN_COMPLETION_MODEL | 5 | 5 | COMPLETE | 25.3 |
| student_por | XGBoost | TRAIN_COMPLETION_MODEL | 5 | 5 | COMPLETE | 34.1 |
| oulad | Logistic Regression | TRAIN_COMPLETION_MODEL | 3 | 5 | COMPLETE | 99.3 |
| oulad | Decision Tree | TRAIN_COMPLETION_MODEL | 3 | 5 | COMPLETE | 107.3 |
| oulad | Random Forest | TRAIN_COMPLETION_MODEL | 3 | 5 | COMPLETE | 624.5 |
| oulad | HistGradientBoosting | TRAIN_COMPLETION_MODEL | 3 | 5 | COMPLETE | 317.9 |
| oulad | SVM | TRAIN_COMPLETION_MODEL | 3 | 5 | COMPLETE | 1580.0 |
| oulad | XGBoost | TRAIN_COMPLETION_MODEL | 3 | 5 | COMPLETE | 166.1 |

## Paired bootstrap

| Dataset | Comparator | CNN-BiLSTM Macro-F1 | Comparator Macro-F1 | Delta | 95% CI | Verdict |
|---|---|---:|---:|---:|---|---|
| student_mat | cnn_only | 0.9015 | 0.8708 | 0.0307 | [0.0042, 0.0580] | CNN_BILSTM_HIGHER |
| student_mat | bilstm_only | 0.9015 | 0.8397 | 0.0617 | [0.0296, 0.0979] | CNN_BILSTM_HIGHER |
| student_mat | logistic_regression | 0.9015 | 0.8793 | 0.0221 | [-0.0023, 0.0473] | PRACTICAL_TIE |
| student_mat | decision_tree | 0.9015 | 0.9067 | -0.0052 | [-0.0213, 0.0119] | PRACTICAL_TIE |
| student_mat | random_forest | 0.9015 | 0.9014 | 0.0001 | [-0.0199, 0.0213] | PRACTICAL_TIE |
| student_mat | hist_gradient_boosting | 0.9015 | 0.8785 | 0.0229 | [-0.0010, 0.0487] | PRACTICAL_TIE |
| student_mat | svm | 0.9015 | 0.8143 | 0.0872 | [0.0490, 0.1268] | CNN_BILSTM_HIGHER |
| student_mat | xgboost | 0.9015 | 0.8880 | 0.0135 | [-0.0066, 0.0353] | PRACTICAL_TIE |
| student_por | cnn_only | 0.8623 | 0.8468 | 0.0155 | [0.0025, 0.0293] | CNN_BILSTM_HIGHER |
| student_por | bilstm_only | 0.8623 | 0.7843 | 0.0780 | [0.0507, 0.1056] | CNN_BILSTM_HIGHER |
| student_por | logistic_regression | 0.8623 | 0.8205 | 0.0417 | [0.0150, 0.0679] | CNN_BILSTM_HIGHER |
| student_por | decision_tree | 0.8623 | 0.8487 | 0.0135 | [-0.0119, 0.0387] | PRACTICAL_TIE |
| student_por | random_forest | 0.8623 | 0.8692 | -0.0070 | [-0.0291, 0.0151] | PRACTICAL_TIE |
| student_por | hist_gradient_boosting | 0.8623 | 0.8506 | 0.0116 | [-0.0165, 0.0397] | PRACTICAL_TIE |
| student_por | svm | 0.8623 | 0.7825 | 0.0798 | [0.0424, 0.1189] | CNN_BILSTM_HIGHER |
| student_por | xgboost | 0.8623 | 0.8664 | -0.0041 | [-0.0276, 0.0198] | PRACTICAL_TIE |
| oulad | cnn_only | 0.8281 | 0.8204 | 0.0077 | [0.0040, 0.0116] | CNN_BILSTM_HIGHER |
| oulad | bilstm_only | 0.8281 | 0.8273 | 0.0008 | [-0.0028, 0.0045] | PRACTICAL_TIE |
| oulad | logistic_regression | 0.8281 | 0.8247 | 0.0033 | [-0.0006, 0.0074] | PRACTICAL_TIE |
| oulad | decision_tree | 0.8281 | 0.8061 | 0.0220 | [0.0169, 0.0272] | CNN_BILSTM_HIGHER |
| oulad | random_forest | 0.8281 | 0.8220 | 0.0061 | [0.0020, 0.0107] | CNN_BILSTM_HIGHER |
| oulad | hist_gradient_boosting | 0.8281 | 0.8241 | 0.0040 | [-0.0001, 0.0082] | PRACTICAL_TIE |
| oulad | svm | 0.8281 | 0.8250 | 0.0031 | [-0.0005, 0.0068] | PRACTICAL_TIE |
| oulad | xgboost | 0.8281 | 0.8259 | 0.0022 | [-0.0019, 0.0064] | PRACTICAL_TIE |

## Resources

The largest directly observed live OULAD worker RSS was 1.14 GiB. Measurement status: `LOWER_BOUND_FROM_PERIODIC_GET_PROCESS_SAMPLING`. Classical comparators ran on CPU, with one RBF-SVM job at a time.

## Integrity and claim boundaries

- Official CNN-BiLSTM checkpoints, predictions, thresholds, and registry selection were not changed.
- The recommendation artifacts and verdict remain unchanged.
- Future OULAD remains `LOCKED_NOT_EXECUTED`.
- Comparator results do not replace the official model selection.
- Full provenance is stored in `artifacts/final/comparator_completion/`.
