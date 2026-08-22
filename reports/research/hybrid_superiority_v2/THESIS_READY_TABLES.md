# Thesis-ready tables (research only — not serving authority)

Protocol `hybrid_superiority_v2.0` hash `eb5f4cfbf4e1…`. Primary = AP (`sklearn.metrics.average_precision_score`). Outer test unused.

## UCI Combined — baseline lock 3×3 mean AP

N=1044, groups=662, prevalence=0.220. Roster gồm XGB và CatBoost.

| Model | S0 | S1 | S2 |
|---|---:|---:|---:|
| CatBoost | **0.501** | **0.769** | **0.907** |
| RF | 0.486 | 0.721 | 0.905 |
| XGB | 0.455 | 0.743 | 0.899 |
| LR | 0.465 | 0.745 | 0.876 |
| SVM | 0.438 | 0.738 | 0.884 |
| DT | 0.447 | 0.735 | 0.884 |
| MLP | 0.442 | 0.702 | 0.840 |

## UCI Hybrid C0-R 3×3 (9 runs)

| | S0 | S1 | S2 |
|---|---:|---:|---:|
| Mean AP | 0.461 | 0.811 | 0.913 |
| Std | 0.047 | 0.043 | 0.025 |
| vs CatBoost | −0.040 | +0.041 | +0.006 |
| Material | cold guardrail 0.05 pass | pass (cần 0.023) | **fail** (cần 0.010) |

Screen fold-0 J: C0-R −3.15 > C3-G −5.12 > C1-R −5.88 > C2-S −7.81.

## OULAD risk-set — SPEED lock fold 0 × seeds 42/1201

N: 20% 26697 / 35% 25606 / 50% 24599 / 75% 23159 / 100% 22522 (Withdrawn còn 94).

| Model | 20pct | 35pct | 50pct | 75pct | 100pct |
|---|---:|---:|---:|---:|---:|
| LR | **0.768** | **0.809** | 0.856 | **0.899** | 0.924 |
| XGB | 0.766 | 0.806 | **0.856** | 0.898 | **0.926** |
| CatBoost | 0.767 | 0.807 | 0.856 | 0.898 | 0.922 |
| SVM | 0.766 | 0.803 | 0.855 | 0.896 | 0.924 |
| MLP | 0.766 | 0.808 | 0.853 | 0.898 | 0.923 |
| RF | 0.748 | 0.789 | 0.849 | 0.894 | 0.922 |
| DT | 0.698 | 0.756 | 0.808 | 0.856 | 0.891 |

## OULAD Hybrid C0-R (3 seed, fold 0, SPEED 10 epoch)

| | 20pct | 35pct | 50pct | 75pct | 100pct |
|---|---:|---:|---:|---:|---:|
| C0-R mean | 0.761 | 0.809 | 0.858 | 0.897 | 0.923 |
| Ceiling | 0.768 | 0.809 | 0.856 | 0.899 | 0.926 |
| Δ | −0.007 | +0.000 | +0.001 | −0.002 | −0.003 |

Không vượt material. Không viết “vượt trội”.
