# Phase 3 — Optuna Results

## Control

| Fold | Macro-F1 | Worst F1 | PR-AUC | NLL | Brier | ECE | Epoch |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 0.768928 | 0.699892 | 0.823836 | 0.462818 | 0.152151 | 0.037270 | 8 |
| 1 | 0.771485 | 0.702466 | 0.827273 | 0.453916 | 0.148496 | 0.028582 | 11 |
| 2 | 0.771248 | 0.703371 | 0.822701 | 0.454839 | 0.148897 | 0.025055 | 8 |

## Selected trials

| Fold | Trial | Macro-F1 | Worst F1 | PR-AUC | NLL | Epoch |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 23 | 0.774114 | 0.707442 | 0.831996 | 0.448519 | 6 |
| 1 | 12 | 0.776996 | 0.710022 | 0.831678 | 0.445320 | 6 |
| 2 | 15 | 0.775243 | 0.709929 | 0.828261 | 0.448700 | 5 |

Search-seed mean Macro-F1: control 0.770554,
selected 0.775451, delta
+0.004897 (SMALL).

Tie-breaking used primary tolerance 1e-4, then worst-stage F1, PR-AUC, NLL,
Brier, and trial number. No outer result entered ranking.
