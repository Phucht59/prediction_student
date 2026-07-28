TECHNICAL EVIDENCE ONLY — MLP IS ONE COMPARATOR IN THE UNIFIED MODEL COMPARISON.

# Standalone MLP Comparator

MLP is a standalone tabular baseline. It is not a CNN-BiLSTM variant. Hyperparameters were selected on inner folds only, and probabilities were averaged across all five registered seeds.

| Dataset | Macro-F1 | Balanced Accuracy | PR-AUC | ECE |
|---|---:|---:|---:|---:|
| student_mat | 0.8595 | 0.8621 | 0.9503 | 0.0797 |
| student_por | 0.8304 | 0.8190 | 0.9147 | 0.0475 |
| oulad | 0.8283 | 0.8219 | 0.8917 | 0.0060 |

## Paired bootstrap: CNN-BiLSTM minus MLP

| Dataset | Delta Macro-F1 | 95% CI | Interpretation | Unit | Replicates |
|---|---:|---|---|---|---:|
| student_mat | 0.0420 | [0.0148, 0.0709] | CNN_BILSTM_HIGHER | record_id | 5000 |
| student_por | 0.0319 | [0.0061, 0.0587] | CNN_BILSTM_HIGHER | record_id | 5000 |
| oulad | -0.0002 | [-0.0036, 0.0031] | INSUFFICIENT_EVIDENCE_OF_DIFFERENCE | id_student | 5000 |

A confidence interval crossing zero is reported as insufficient evidence of a difference, not as equivalence.
