# Experiment Protocol

## Final comparator completion

The complete nine-model release matrix is governed by
`configs/final/comparator_completion_protocol.yaml` and
`docs/COMPARATOR_COMPLETION_PROTOCOL.md`. The protocol was committed before
any completion-model training.

Only Student-Mat/Student-Por XGBoost and the six unified OULAD classical-ML
comparators may be trained by this completion. All official CNN-BiLSTM
checkpoints, predictions, thresholds, and selections remain immutable.
Future OULAD remains `LOCKED_NOT_EXECUTED`.

Comparator hyperparameters are selected exclusively by inner CV. Outer folds
produce record-aligned OOF probabilities, and final metrics use the arithmetic
mean probability across all registered seeds. OULAD thresholds are fit from
inner-OOF ensemble probabilities separately for each outer fold.

This release performs no experiment or training. It reuses registered frozen outer-fold predictions and probability ensembles. Hyperparameter screening, inner validation, best-fold results and best-seed results are excluded from final tables. Seed stability may be reported separately but is never relabeled as an ensemble.

Future OULAD is `LOCKED_NOT_EXECUTED`. The final recommendation audit is technical; expert-label metrics remain unavailable until labels are independently supplied.
