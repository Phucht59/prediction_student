# Teacher Feedback Completion

| Requirement | Evidence | Status |
|---|---|---|
| G3 is final grade 0-20 | `artifacts/final/teacher_feedback_validation/uci_target_contract.json` | PASS |
| Low 0-9 | `artifacts/final/teacher_feedback_validation/uci_target_contract.json` | PASS |
| Medium 10-14 | `artifacts/final/teacher_feedback_validation/uci_target_contract.json` | PASS |
| High 15-20 | `artifacts/final/teacher_feedback_validation/uci_target_contract.json` | PASS |
| G3 excluded from predictors | `artifacts/final/uci_timing_scenarios/leakage_validation.json` | PASS |
| S0 no G1/G2 | `artifacts/final/uci_timing_scenarios/leakage_validation.json` | PASS |
| S1 G1 only | `artifacts/final/uci_timing_scenarios/leakage_validation.json` | PASS |
| S2 G1+G2 | `artifacts/final/uci_timing_scenarios/leakage_validation.json` | PASS |
| MLP MAT | `artifacts/final/teacher_feedback_validation/mlp_comparator/student_mat/` | PASS |
| MLP POR | `artifacts/final/teacher_feedback_validation/mlp_comparator/student_por/` | PASS |
| MLP OULAD | `artifacts/final/teacher_feedback_validation/mlp_comparator/oulad/` | PASS |
| same outer splits | `artifacts/final/teacher_feedback_validation/split_equivalence.json` | PASS |
| outer not used for tuning | `artifacts/final/teacher_feedback_validation/evaluation_contract.json` | PASS |
| train-only preprocessing | `artifacts/final/teacher_feedback_validation/evaluation_contract.json` | PASS |
| invalid ADASYN/SMOTE removed | `artifacts/final/teacher_feedback_validation/imbalance_safety_audit.json` | PASS |
| synthetic oversampling OULAD tensor | `artifacts/final/teacher_feedback_validation/imbalance_safety_audit.json` | ABSENT |
| CNN-vs-MLP bootstrap MAT | `artifacts/final/teacher_feedback_validation/paired_bootstrap_cnn_bilstm_vs_mlp.csv` | PASS |
| CNN-vs-MLP bootstrap POR | `artifacts/final/teacher_feedback_validation/paired_bootstrap_cnn_bilstm_vs_mlp.csv` | PASS |
| CNN-vs-MLP bootstrap OULAD | `artifacts/final/teacher_feedback_validation/paired_bootstrap_cnn_bilstm_vs_mlp.csv` | PASS |
| xAPI absent | `artifacts/final/teacher_feedback_validation/regression_guard_after.json` | PASS |
| Future OULAD locked | `artifacts/final/teacher_feedback_validation/evaluation_contract.json` | PASS |
| expert labels pending | `artifacts/final/recommendation/expert_evaluation/expert_metrics.json` | EXPECTED |
| DEEP_TIMING_DIAGNOSTIC | `artifacts/final/teacher_feedback_validation/deep_timing_feasibility.json` | NOT_RUN_ARCHITECTURE_NOT_COMPARABLE |

Official CNN-BiLSTM selection, checkpoints, headline metrics, recommendation counts, and expert-label status remain frozen.

The historical unsafe plain-SMOTE/ADASYN UCI baseline path is disclosed in the imbalance audit and is not used by this completion evidence. Safe S2 classical revalidation uses no synthetic resampling.

Deep timing is intentionally not run. The frozen UCI hybrid requires exactly two grade timesteps and has no explicit availability mask; manufacturing S0/S1 inputs would not be an architecture-comparable test. The standalone MLP study is the registered information-availability diagnostic.
