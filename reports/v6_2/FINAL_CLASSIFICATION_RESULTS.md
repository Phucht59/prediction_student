# Canonical final classification results

These are consolidated references to existing frozen OOF evidence. V6.2
does not retrain, tune, or replace any prediction model.

| Dataset | Role | Model | Macro-F1 | Balanced accuracy | PR-AUC | Risk F1 | Brier |
|---|---|---|---:|---:|---:|---:|---:|
| student_mat | OFFICIAL_CNN_BILSTM | CNN-BiLSTM | 0.9015 | 0.9021 | 0.9442 |  | 0.2072 |
| student_mat | KEY_ABLATION | CNN-only | 0.8708 | 0.8778 | 0.9300 |  | 0.2278 |
| student_mat | KEY_ABLATION | BiLSTM-only | 0.8397 | 0.8517 | 0.8950 |  | 0.3069 |
| student_mat | STRONGEST_CLASSICAL_BY_MACRO_F1 | Decision Tree | 0.9067 | 0.9041 | 0.8609 |  | 0.1816 |
| student_por | OFFICIAL_CNN_BILSTM | CNN-BiLSTM | 0.8623 | 0.8676 | 0.9147 |  | 0.1725 |
| student_por | KEY_ABLATION | CNN-only | 0.8468 | 0.8518 | 0.9215 |  | 0.1686 |
| student_por | KEY_ABLATION | BiLSTM-only | 0.7843 | 0.7986 | 0.8649 |  | 0.2384 |
| student_por | STRONGEST_CLASSICAL_BY_MACRO_F1 | Random Forest | 0.8692 | 0.8836 | 0.9309 |  | 0.1569 |
| oulad | OFFICIAL_CNN_BILSTM | CNN-BiLSTM | 0.8281 | 0.8203 | 0.8934 | 0.7826 | 0.1134 |
| oulad | KEY_ABLATION | CNN-only | 0.8204 | 0.8124 | 0.8884 | 0.7722 | 0.1170 |
| oulad | KEY_ABLATION | BiLSTM-only | 0.8273 | 0.8223 | 0.8904 | 0.7855 | 0.1158 |
| oulad | STRONGEST_CLASSICAL_BY_MACRO_F1 | XGBoost | 0.8259 | 0.8222 | 0.8900 | 0.7855 | 0.1159 |
