# One-weight baselines on Hybrid tensors

Protocol: inner 3 fold × 3 seed (`42`, `1201`, `2026`). One fitted estimator per family scores every stage. Features = Hybrid tensors (static + aggregate + masked temporal + progress). **No** last/mean/max/std/slope. Outer test: false.

Hybrid numbers in this file are the **same-run** one-checkpoint model used as the architecture comparator (not the serving tables in `uci_final.csv` / `oulad_final.csv`). AP = `sklearn.metrics.average_precision_score`.

XGB and CatBoost are **not** on the serving roster. They are the tensor-parity ceiling.

Source: `artifacts/research/hybrid_superiority_v2/metrics/baseline_fair_stage_metrics_{uci,oulad}.csv`, `headline_hybrid_{uci,oulad}.csv`. Mean of 9 runs.

## UCI — AP

| Model | S0 | S1 | S2 |
|---|---:|---:|---:|
| Hybrid CNN–BiLSTM | 0.4559 | **0.8110** | **0.9132** |
| LR | 0.4234 | 0.7687 | 0.8955 |
| RF | **0.4796** | 0.7080 | 0.8494 |
| XGB | 0.4503 | 0.7278 | 0.8469 |
| CatBoost | 0.4463 | 0.7090 | 0.8199 |
| SVM | 0.4299 | 0.7390 | 0.7593 |
| DT | 0.4298 | 0.6700 | 0.7667 |
| MLP | 0.4337 | 0.5381 | 0.6460 |

## UCI — F1 at STOP `t`

| Model | S0 | S1 | S2 |
|---|---:|---:|---:|
| Hybrid CNN–BiLSTM | 0.4149 | 0.7180 | **0.7975** |
| LR | 0.4173 | 0.6671 | 0.7953 |
| RF | **0.5296** | 0.7399 | 0.7706 |
| XGB | 0.4833 | **0.7453** | 0.7914 |
| CatBoost | 0.4916 | 0.7254 | 0.7625 |
| DT | 0.4853 | 0.7270 | 0.7472 |
| SVM | 0.4689 | 0.6244 | 0.6417 |
| MLP | 0.4435 | 0.5284 | 0.5705 |

## OULAD — AP

| Model | 20 | 35 | 50 | 75 | 100 |
|---|---:|---:|---:|---:|---:|
| Hybrid CNN–BiLSTM | 0.7469 | **0.8054** | **0.8524** | 0.8929 | **0.9190** |
| XGB | **0.7661** | 0.8027 | 0.8512 | **0.8935** | 0.9187 |
| CatBoost | 0.7618 | 0.7988 | 0.8476 | 0.8900 | 0.9143 |
| LR | 0.7556 | 0.7958 | 0.8415 | 0.8853 | 0.9166 |
| RF | 0.7536 | 0.7920 | 0.8448 | 0.8877 | 0.9129 |
| SVM | 0.7291 | 0.7832 | 0.8250 | 0.8767 | 0.9050 |
| DT | 0.7049 | 0.7459 | 0.7984 | 0.8476 | 0.8745 |
| MLP | 0.7025 | 0.7539 | 0.8052 | 0.8645 | 0.9068 |

## OULAD — F1 at STOP `t`

| Model | 20 | 35 | 50 | 75 | 100 |
|---|---:|---:|---:|---:|---:|
| Hybrid CNN–BiLSTM | 0.6741 | **0.7008** | 0.7324 | **0.7919** | **0.8366** |
| XGB | **0.6833** | 0.6986 | **0.7352** | 0.7890 | 0.8360 |
| CatBoost | 0.6797 | 0.6937 | 0.7307 | 0.7877 | 0.8293 |
| LR | 0.6746 | 0.6990 | 0.7266 | 0.7813 | 0.8292 |
| RF | 0.6740 | 0.6880 | 0.7287 | 0.7810 | 0.8294 |
| SVM | 0.6612 | 0.6830 | 0.7104 | 0.7725 | 0.8161 |
| DT | 0.6445 | 0.6535 | 0.6949 | 0.7513 | 0.8014 |
| MLP | 0.6480 | 0.6609 | 0.6988 | 0.7615 | 0.8203 |

## Reading

- UCI S1/S2: Hybrid leads AP (S1 +0.042 vs LR; S2 +0.018 vs LR). S0: RF 0.480 vs Hybrid 0.456 — no grade sequence, CNN/BiLSTM off.
- OULAD 20%: XGB 0.766 vs Hybrid 0.747. From 35% to 100%, |Δ AP| Hybrid vs XGB ≤ 0.003 (tie within run noise).
- Do not claim architecture superiority on OULAD against this ceiling. Serving model stays Hybrid CNN–BiLSTM.
- Gate vs this ceiling: not passed. Outer test not opened.
