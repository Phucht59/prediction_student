# Fair one-weight baselines

Each baseline family is **one fitted estimator** scoring every stage (same rule as Hybrid C0-R).
That single weight set is **Optuna-best** on stacked warm-macro AP, protocol search space,
budget 40 UCI / 28 OULAD trials (not a 4-trial skim).
Per-stage XGB/CatBoost envelope remains a diagnostic, not the scientific comparator.

Outer test: **false**. Serving Hybrid: **unchanged**.

## UCI
| Mốc | Fair family (one weight) | Fair AP | C0-R | Δ |
|---|---:|---:|---:|---:|
| S0 | RF | 0.4891 | 0.4612 | -0.0279 |
| S1 | LR | 0.7568 | 0.8106 | +0.0537 |
| S2 | LR | 0.8897 | 0.9128 | +0.0231 |

Family mean AP (one model per family):

| family | S0 | S1 | S2 |
|---|---:|---:|---:|
| CatBoost | 0.4335 | 0.7224 | 0.8090 |
| DT | 0.4148 | 0.7409 | 0.7678 |
| LR | 0.4187 | 0.7568 | 0.8897 |
| MLP | 0.4279 | 0.6707 | 0.8310 |
| RF | 0.4891 | 0.7173 | 0.8406 |
| SVM | 0.3893 | 0.7370 | 0.8313 |
| XGB | 0.4417 | 0.7348 | 0.8557 |

Skipped under deadline: `[]`

## OULAD
| Mốc | Fair family (one weight) | Fair AP | C0-R | Δ |
|---|---:|---:|---:|---:|
| 20pct | XGB | 0.7666 | 0.7476 | -0.0190 |
| 35pct | CatBoost | 0.8055 | 0.8056 | +0.0001 |
| 50pct | CatBoost | 0.8535 | 0.8522 | -0.0014 |
| 75pct | XGB | 0.8966 | 0.8929 | -0.0037 |
| 100pct | XGB | 0.9235 | 0.9191 | -0.0044 |

Family mean AP (one model per family):

| family | 100pct | 20pct | 35pct | 50pct | 75pct |
|---|---:|---:|---:|---:|---:|
| CatBoost | 0.9205 | 0.7665 | 0.8055 | 0.8535 | 0.8958 |
| DT | 0.8883 | 0.7125 | 0.7563 | 0.8129 | 0.8608 |
| LR | 0.9164 | 0.7582 | 0.7995 | 0.8474 | 0.8892 |
| MLP | 0.9213 | 0.7595 | 0.7998 | 0.8496 | 0.8941 |
| RF | 0.9214 | 0.7534 | 0.7960 | 0.8487 | 0.8931 |
| SVM | 0.9131 | 0.7347 | 0.7969 | 0.8457 | 0.8894 |
| XGB | 0.9235 | 0.7666 | 0.8039 | 0.8533 | 0.8966 |

Skipped under deadline: `[]`

## Gate vs fair ceiling (uci)

pass=`True` cold_ok=`True` warm_fail=`0`

## Gate vs fair ceiling (oulad)

pass=`False` cold_ok=`True` warm_fail=`4`

## Decision

NOT_READY_FOR_DEFENSE

Do not write vượt trội unless both fair-ceiling gates pass. Do not open outer test.
