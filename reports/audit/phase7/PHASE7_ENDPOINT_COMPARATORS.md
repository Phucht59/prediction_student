# Phase 7 Endpoint Comparators

Protocol-matched endpoint metrics:

| Model | Macro-F1 | PR-AUC | ROC-AUC | NLL | Brier | ECE |
|---|---:|---:|---:|---:|---:|---:|
| H1 tabular residual | 0.798400 | 0.863039 | 0.876142 | 0.406292 | 0.130702 | 0.011988 |
| H0 CNN-BiLSTM | 0.828084 | 0.893355 | 0.908156 | 0.358778 | 0.113355 | 0.009463 |
| MLP | 0.828286 | 0.891710 | 0.907336 | 0.362018 | 0.114346 | 0.007746 |

Historical comparator predictions were reused only after exact endpoint record,
fold and target identity checks. The H1 result comes from the new frozen
Phase 7 checkpoints. This table must not be mixed with mean-stage
early-warning metrics.
