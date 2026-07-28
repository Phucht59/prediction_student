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
| Target and timing separated | `artifacts/final/uci_timing_scenarios/leakage_validation.json` | PASS |
| G1/G2 band relationship | `artifacts/final/uci_timing_scenarios/grade_band_relationship.json` | PASS |
| All 10 models × S0/S1/S2 × MAT/POR | `artifacts/final/uci_timing_scenarios/scenario_rankings.csv` | PASS (60/60) |
| Fair timing uses no transfer | `artifacts/final/teacher_feedback_validation/fair_comparison_contract.json` | PASS |
| Same folds/raw information/preprocessing/metrics | `artifacts/final/teacher_feedback_validation/fair_comparison_contract.json` | PASS |
| Low/Medium/High metrics complete | `artifacts/final/uci_timing_scenarios/per_class_metrics.csv` | PASS (180/180) |
| Mask-aware deep timing | `artifacts/final/uci_timing_scenarios/deep_mask_validation.json` | PASS |
| Unified paired bootstrap | `artifacts/final/uci_timing_scenarios/paired_bootstrap_all_models.csv` | PASS |
| Final imbalance safety | `artifacts/final/teacher_feedback_validation/imbalance_final_safety_audit.json` | PASS |
| OULAD 47-channel temporal audit | `artifacts/final/teacher_feedback_validation/oulad_temporal_branch_audit.json` | PASS |

Official CNN-BiLSTM selection, checkpoints, headline metrics, recommendation counts, and expert-label status remain frozen.

The historical unsafe plain-SMOTE/ADASYN UCI baseline path is disclosed in the imbalance audit and is not used by this completion evidence. Safe S2 classical revalidation uses no synthetic resampling.

The fair diagnostic uses a separate mask-aware implementation with one fixed
`[batch, 2, 7]` contract across S0/S1/S2. It uses no transfer or official
checkpoint and does not replace the frozen official Student-Mat/Student-Por
models. MLP is one member of the unified ten-model matrix, not a separate
scientific study.
