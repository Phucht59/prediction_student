# Phase 7 Endpoint Tuning

The architecture was frozen as `H1_TABULAR_RESIDUAL_EXPERT`; only training
policy fields in the registered search space were eligible.

| Item | Value |
|---|---:|
| Scheduled trials | 54 |
| Completed trials | 27 |
| Pruned trials | 27 |
| Failed trials | 0 |
| Architecture hash count | 1 |
| Parameter count | 160,492 |

The best trial from each outer-train partition was tested against its CONTROL
under two predefined stability seeds. The preregistered selection rule chose
CONTROL because tuned mean Macro-F1 was lower by 0.000073. No outer labels
entered trial scoring, pruning or selection.
