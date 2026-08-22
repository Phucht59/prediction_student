# C0-R scientific evaluation

Topology **C0-R** (Hybrid CNN–BiLSTM). C4-STAR is in `test_lab/` and is not the thesis model.

Outer test: **false**. Serving Hybrid: **unchanged**. Combined gate: **FAIL**. Status: **NOT_READY_FOR_DEFENSE**.

## What this run added

OULAD C0-R is now measured **3 fold × 3 seed** against the v2.1 XGB/LR ceiling (not SPEED). That was the missing scientific piece.

UCI already had a stronger 3×3 (24-trial HPO + 9 runs: S1 0.811 / S2 0.913). An extra 8-trial HPO **did not beat it** (S1 0.790 / S2 0.900, S0 cold fail). **UCI authority stays the original 9-run table.**

## UCI — keep original C0-R 3×3 vs CatBoost

| Mốc | CatBoost | C0-R 3×3 | Δ | Gate |
|---|---:|---:|---:|---|
| S0 | 0.501 | 0.461 | −0.040 | cold pass (≤0.05) |
| S1 | 0.769 | **0.811** | **+0.041** | **material pass** (cần +0.023) |
| S2 | 0.907 | **0.913** | +0.006 | **material fail** (cần +0.010) |

8-trial re-HPO (not used as authority): S0 0.411 / S1 0.790 / S2 0.900.

## OULAD — new C0-R 3×3 vs v2.1 ceiling

| Mốc | Trần | C0-R 3×3 | Δ | Gate |
|---|---:|---:|---:|---|
| 20% | LR 0.768 | 0.748 | −0.020 | cold **fail** (guardrail 0.02, sát ngưỡng) |
| 35% | XGB 0.808 | 0.806 | −0.002 | thua |
| 50% | XGB 0.855 | 0.852 | −0.002 | thua |
| 75% | XGB 0.897 | 0.893 | −0.004 | thua |
| 100% | XGB 0.924 | 0.919 | −0.005 | thua |

HPO fold 0 had one trial J>0 (100% AP 0.926); **3×3 does not hold**.

## Decision

Do not write vượt trội. Do not open outer test. C0-R remains the Hybrid to defend as *architecture*, with an honest UCI S2 miss of ~0.004 and OULAD slightly below XGB on a full 3×3.
