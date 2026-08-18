# Final authority imbalance policy audit

## Policy distinction
- **AUTHORITY_POLICY** reproduces the frozen authority exactly, including any fold-specific historical choices.
- **CONTROLLED_FIXED_MODE** is a counterfactual with one forced mode across every fold. It is not expected to equal an authority whose policy was mixed.

## Per-fold authority policy

| dataset | outer fold | effective strategy | class/positive weight | sampler / resampling | source |
|---|---:|---|---|---|---|
| student-mat | 0 | standard_cross_entropy | none (SharedTrunkSubjectHeadsV51 calls multitask_loss(class_weights=None)) | none / none | `historical_uci_v5_1/artifacts/v5_1/student_mat/selected_configs.json; historical_uci_v5_1/src/studies/v5_1/uci/runner.py` @ `f51e1f43d9af768194bb34ea5ff4aeb8e4cbd502` |
| student-mat | 1 | focal_loss | none (SharedTrunkSubjectHeadsV51 calls multitask_loss(class_weights=None)) | none / none | `historical_uci_v5_1/artifacts/v5_1/student_mat/selected_configs.json; historical_uci_v5_1/src/studies/v5_1/uci/runner.py` @ `f51e1f43d9af768194bb34ea5ff4aeb8e4cbd502` |
| student-mat | 2 | focal_loss | none (SharedTrunkSubjectHeadsV51 calls multitask_loss(class_weights=None)) | none / none | `historical_uci_v5_1/artifacts/v5_1/student_mat/selected_configs.json; historical_uci_v5_1/src/studies/v5_1/uci/runner.py` @ `f51e1f43d9af768194bb34ea5ff4aeb8e4cbd502` |
| student-mat | 3 | focal_loss | none (SharedTrunkSubjectHeadsV51 calls multitask_loss(class_weights=None)) | none / none | `historical_uci_v5_1/artifacts/v5_1/student_mat/selected_configs.json; historical_uci_v5_1/src/studies/v5_1/uci/runner.py` @ `f51e1f43d9af768194bb34ea5ff4aeb8e4cbd502` |
| student-mat | 4 | focal_loss | none (SharedTrunkSubjectHeadsV51 calls multitask_loss(class_weights=None)) | none / none | `historical_uci_v5_1/artifacts/v5_1/student_mat/selected_configs.json; historical_uci_v5_1/src/studies/v5_1/uci/runner.py` @ `f51e1f43d9af768194bb34ea5ff4aeb8e4cbd502` |
| student-por | 0 | none | none | none / none | `historical_uci_v5_1/artifacts/v5_1/student_por/selected_configs.json; historical_uci_v5_1/src/studies/v5_1/uci/runner.py` @ `f51e1f43d9af768194bb34ea5ff4aeb8e4cbd502` |
| student-por | 1 | class_weight | inverse_frequency_from_outer_train | none / none | `historical_uci_v5_1/artifacts/v5_1/student_por/selected_configs.json; historical_uci_v5_1/src/studies/v5_1/uci/runner.py` @ `f51e1f43d9af768194bb34ea5ff4aeb8e4cbd502` |
| student-por | 2 | none | none | none / none | `historical_uci_v5_1/artifacts/v5_1/student_por/selected_configs.json; historical_uci_v5_1/src/studies/v5_1/uci/runner.py` @ `f51e1f43d9af768194bb34ea5ff4aeb8e4cbd502` |
| student-por | 3 | none | none | none / none | `historical_uci_v5_1/artifacts/v5_1/student_por/selected_configs.json; historical_uci_v5_1/src/studies/v5_1/uci/runner.py` @ `f51e1f43d9af768194bb34ea5ff4aeb8e4cbd502` |
| student-por | 4 | focal | none | none / none | `historical_uci_v5_1/artifacts/v5_1/student_por/selected_configs.json; historical_uci_v5_1/src/studies/v5_1/uci/runner.py` @ `f51e1f43d9af768194bb34ea5ff4aeb8e4cbd502` |
| oulad | 0 | none | none | none / none | `artifacts/final_candidate_freeze/FINAL_H1_FREEZE_MANIFEST.json training_policy.per_outer_fold_config` @ `2c8baa96f563d7a4a5188abfd0c700828a91e301` |
| oulad | 1 | none | none | none / none | `artifacts/final_candidate_freeze/FINAL_H1_FREEZE_MANIFEST.json training_policy.per_outer_fold_config` @ `2c8baa96f563d7a4a5188abfd0c700828a91e301` |
| oulad | 2 | none | none | none / none | `artifacts/final_candidate_freeze/FINAL_H1_FREEZE_MANIFEST.json training_policy.per_outer_fold_config` @ `2c8baa96f563d7a4a5188abfd0c700828a91e301` |

## Findings
- Student-Mat: MIXED_EFFECTIVE_LOSS (folds 0 standard CE; 1-4 focal loss; no class weights, sampler, or resampling in shared-subject trainer). The historical selected imbalance label is not consumed by `fit_shared_subject_model`; its actual class-weight argument is `None`.
- Student-Por: MIXED (fold0 NONE; fold1 CLASS_WEIGHT; folds2-3 NONE; fold4 FOCAL). Outer fold 1 is the only CLASS_WEIGHT fold. It requires five authority-policy replay jobs; 20 completed POR_FIXED_NONE jobs can be retained separately, but cannot be reused in the authority ensemble.
- OULAD: UNIFORM_NONE (standard BCE in every fold; no class/positive weight, sampler, or resampling). The frozen FINAL H1 is `H1_TABULAR_RESIDUAL_EXPERT` with 160492 parameters, three outer folds, five seeds, and no resampling/sampler/positive weighting.

## Decision
The controlled fixed-mode experiment is not yet valid to launch. First run and verify only the five POR fold-1 authority-policy replay jobs; then implement and validate fixed CLASS_WEIGHT/SMOTE/ADASYN semantics without modifying the frozen authority. No training was launched by this audit.
