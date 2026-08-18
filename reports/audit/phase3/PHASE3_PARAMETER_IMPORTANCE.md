# Phase 3 — Parameter Importance

| Rank | Variable | Mean importance |
| ---: | --- | ---: |
| 1 | loss_policy | 0.2928 |
| 2 | learning_rate | 0.2474 |
| 3 | outcome_weight | 0.1428 |
| 4 | survival_weight | 0.1277 |
| 5 | dropout | 0.0933 |
| 6 | batch_size | 0.0845 |
| 7 | weight_decay | 0.0116 |

These values are **SEARCH ASSOCIATIONS**, not causal effects. Conditional
positive-weight strategy is not common to all trials and is therefore absent
from Optuna's common-parameter importance output.
