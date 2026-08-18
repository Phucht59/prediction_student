# Final H1 Stage Results

| Model | 20% | 35% | 50% | 75% | Mean | Worst |
| --- | --- | --- | --- | --- | --- | --- |
| MLP | 0.711734 | 0.749499 | 0.795505 | 0.853659 | 0.777599 | 0.711734 |
| H0 Current Hybrid | 0.709888 | 0.744534 | 0.792676 | 0.852419 | 0.774879 | 0.709888 |
| H1 Tabular Residual Hybrid | 0.713635 | 0.750632 | 0.793953 | 0.850333 | 0.777138 | 0.713635 |

| Stage | H1 − MLP | H1 − H0 |
| --- | --- | --- |
| 20% | 0.001901 | 0.003747 |
| 35% | 0.001133 | 0.006098 |
| 50% | -0.001552 | 0.001277 |
| 75% | -0.003326 | -0.002086 |

H1 improves over MLP at 20% and 35%, then is slightly below it at 50% and 75%.
Relative to H0, H1 improves 20%, 35%, and 50%, but regresses by about 0.00209
at 75%. The residual expert therefore generalized mainly at early/middle
stages, not as a uniform late-stage gain.

## H1 confusion and risk metrics

Counts are fold-averaged because the authoritative stage table aggregates the
three outer folds.

| Stage | TN | FP | FN | TP | Risk precision | Risk recall | Specificity |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 20% | 2378.7 | 706.7 | 842.7 | 1594.0 | 0.692879 | 0.654166 | 0.770970 |
| 35% | 2547.3 | 539.0 | 733.3 | 1495.7 | 0.736049 | 0.670644 | 0.825277 |
| 50% | 2690.7 | 396.0 | 597.0 | 1442.3 | 0.785856 | 0.707135 | 0.871663 |
| 75% | 2912.0 | 174.7 | 477.3 | 1314.3 | 0.886700 | 0.733528 | 0.943374 |
