# Phase 7 Endpoint Stability

| Configuration | Mean Macro-F1 | Std | PR-AUC | NLL | Brier | ECE |
|---|---:|---:|---:|---:|---:|---:|
| CONTROL | 0.795913 | 0.001788 | 0.857556 | 0.414219 | 0.133402 | 0.015222 |
| TUNED | 0.795840 | 0.001045 | 0.857700 | 0.413834 | 0.133290 | 0.015076 |

The tuned configuration was slightly better calibrated but did not improve
the primary metric. Its Macro-F1 delta was below zero, so the registered rule
selected CONTROL. Stability used exactly two predefined development seeds
across all three outer-train partitions.
