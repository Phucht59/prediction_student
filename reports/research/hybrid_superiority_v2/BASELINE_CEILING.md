# Baseline ceiling

Protocol `hybrid_superiority_v2.0` hash `eb5f4cfbf4e1…`. Primary = AP. Outer test unused. Roster includes XGB and CatBoost.

## UCI
Folds=[0, 1, 2] seeds=[42, 1201, 2026] trials=40.

Lock: `artifacts/research/hybrid_superiority_v2/runs/baseline_lock_uci.json`

| Model | S0 | S1 | S2 |
| --- | --- | --- | --- |
| CatBoost | **0.5010** | **0.7694** | **0.9067** |
| DT | 0.4466 | 0.7346 | 0.8843 |
| LR | 0.4650 | 0.7448 | 0.8763 |
| MLP | 0.4421 | 0.7021 | 0.8396 |
| RF | 0.4863 | 0.7211 | 0.9048 |
| SVM | 0.4380 | 0.7384 | 0.8843 |
| XGB | 0.4551 | 0.7430 | 0.8990 |

### Material margin (warm only — cold uses guardrail, not this table)

| Stage | Ceiling model | AP_B | MaterialMargin | Hybrid cần |
| --- | --- | --- | --- | --- |
| S1 | CatBoost | 0.7694 | 0.0231 | 0.7925 |
| S2 | CatBoost | 0.9067 | 0.0100 | 0.9167 |

## OULAD
SPEED_FINISH: trials=4 skip=['DT', 'SVM', 'MLP', 'RF'] folds=[0] seeds=[42, 1201].

Lock: `artifacts/research/hybrid_superiority_v2/runs/baseline_lock_oulad.json`

| Model | 20pct | 35pct | 50pct | 75pct | 100pct |
| --- | --- | --- | --- | --- | --- |
| CatBoost | 0.7665 | 0.8070 | 0.8557 | 0.8984 | 0.9223 |
| DT | 0.6975 | 0.7559 | 0.8080 | 0.8563 | 0.8910 |
| LR | **0.7684** | **0.8087** | 0.8559 | **0.8989** | 0.9240 |
| MLP | 0.7661 | 0.8078 | 0.8532 | 0.8985 | 0.9231 |
| RF | 0.7484 | 0.7891 | 0.8494 | 0.8939 | 0.9224 |
| SVM | 0.7657 | 0.8035 | 0.8545 | 0.8965 | 0.9244 |
| XGB | 0.7659 | 0.8057 | **0.8563** | 0.8980 | **0.9260** |

### Material margin (warm only — cold uses guardrail, not this table)

| Stage | Ceiling model | AP_B | MaterialMargin | Hybrid cần |
| --- | --- | --- | --- | --- |
| 35pct | LR | 0.8087 | 0.0191 | 0.8278 |
| 50pct | XGB | 0.8563 | 0.0144 | 0.8706 |
| 75pct | LR | 0.8989 | 0.0101 | 0.9090 |
| 100pct | XGB | 0.9260 | 0.0100 | 0.9360 |
