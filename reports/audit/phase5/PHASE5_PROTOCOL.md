# Phase 5 — Protocol

- Evidence is INNER development only; outer-test labels are unavailable to the runner.
- M0 is the repository-authoritative sklearn MLP `(64, 32)`.
- H0 reproduces the frozen 150,202-parameter A0 CNN-BiLSTM.
- H1 adds only a compact `178→48→32→logit` tabular residual expert with bounded
  learnable alpha initialized at 0.05.
- CNN, BiLSTM, pooling, A0 fusion, stage policy, loss policy, checkpoint objective,
  and pooled-inner-OOF research threshold remain frozen.
- Screening uses seed 42 and all three outer-train partitions.
- Stability uses preregistered seeds 1201 and 2026.
- Distillation uses cross-fitted MLP teacher probabilities and fixed lambdas
  `{0.05, 0.10, 0.20}`.
- No Optuna, outer evaluation, SMOTE/ADASYN, focal loss, or CNN-depth search.
