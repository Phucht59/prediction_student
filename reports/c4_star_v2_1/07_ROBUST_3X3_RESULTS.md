# 07 Robust 3×3

Inner 3 folds × 3 seeds. Outer unused.

## UCI

| Stage | Ceiling | C4 mean | Δ | material | pos | material pass |
|---|---:|---:|---:|---:|---|---|
| S0 | 0.5010 (CatBoost) | 0.4925 | -0.0085 | 0.0500 | True | None |
| S1 | 0.7694 (CatBoost) | 0.7747 | 0.0053 | 0.0231 | True | False |
| S2 | 0.9067 (CatBoost) | 0.8559 | -0.0508 | 0.0100 | None | False |

Gate pass=`False`. Runs=`9`. Mechanism=`M4`.

## OULAD

| Stage | Ceiling | C4 mean | Δ | material | pos | material pass |
|---|---:|---:|---:|---:|---|---|
| 20pct | 0.7678 (LR) | 0.7564 | -0.0114 | 0.0200 | True | None |
| 35pct | 0.8077 (XGB) | 0.8031 | -0.0045 | 0.0192 | None | False |
| 50pct | 0.8545 (XGB) | 0.8521 | -0.0024 | 0.0145 | None | False |
| 75pct | 0.8969 (XGB) | 0.8938 | -0.0032 | 0.0103 | None | False |
| 100pct | 0.9245 (XGB) | 0.9207 | -0.0037 | 0.0100 | None | False |

Gate pass=`False`. Runs=`9`. Mechanism=`M4`.
