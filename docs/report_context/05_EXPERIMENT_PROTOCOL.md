# Experiment protocol

The dataset uses a deterministic stratified 80/20 locked split. Train and test
split hashes are recorded in the selected configuration and final manifest.
All preprocessing, encoding, feature processing and optional resampling are fit
only on the relevant training partition; validation/test partitions transform
only. Oversampling is never performed before a split.

Model selection uses development records only: five outer stratified folds,
three inner folds, 30 Optuna trials per inner search, selection seed 42 and
mean inner-CV Macro-F1 objective. The selection rule uses one fixed seed and no
ensemble selection; the locked test is excluded. The frozen final strategy is
single seed 42, argmax threshold and calibration type `none`.

Locked test is not used in Optuna, hyperparameter tuning, or final CNN–BiLSTM
configuration selection. Baseline and ablation results on locked test are
post-hoc comparisons after the protocol and final configuration were frozen.

The legacy selected-config label `weighted_ce` resolves to **CrossEntropyLoss
without class weighting** because `class_weight_mode` is `none`.

Nested outer performance estimates CNN-BiLSTM at 0.8781 +/- 0.0448 Macro-F1.
Like-for-like nested HGB is 0.8690. HGB 0.8969 is a different train-pool
five-fold baseline protocol and must be shown separately. Locked-test results
are final evaluation only, not a tie-break.
