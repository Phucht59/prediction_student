# Threshold and calibration audit

| Model | Registered Macro-F1 | Macro-F1 @ 0.5 | Diagnostic outer-oracle Macro-F1 |
|---|---:|---:|---:|
| H0 | 0.828084 | 0.826530 | 0.827993 |
| H1 | 0.798400 | 0.796718 | 0.799107 |

The outer-oracle rows are diagnosis only and were not used to change any
threshold or model. H1 can recover less than 0.001 Macro-F1 by a global oracle
threshold, while PR-AUC, ROC-AUC, NLL and Brier all regress. The primary
deficit is ranking/representation/training, not threshold selection.

H0's original report uses 10-bin ECE (0.008683); the Phase 7
aligned comparator uses 15-bin ECE (0.009463). This bin-count
provenance difference does not affect Macro-F1.
