# 02 Data and leakage audit

## UCI

- Target: `G3 < 10`. G3 never a predictor.
- G1/G2 Hybrid path: temporal only. Aggregate disabled.
- Groups: student identity across Math/Portuguese; outer/inner group-disjoint (tests).
- Stages S0/S1/S2 mask G1 then G2.

## OULAD

- Target: Fail|Withdrawn vs Pass|Distinction.
- Operational risk-set per cutoff (primary).
- Events strictly before cutoff (`events_strictly_before_cutoff`).
- Forbidden: final_result, score, date_unregistration as features.
- 100% operational risk-set: 22522 rows, 94 Withdrawn remaining — not an early-warning panel.
- FIT-only scalers.

## Splits

Locked hashes recorded in `00_INPUT_AUDIT.md`. v2.1 **must not** regenerate outer folds.

## Teacher

Cross-fit inside FIT only. STOP/VALID/outer never in teacher training.

## SPEED_FINISH

Not a leakage event, but an under-powered ceiling. v2.1 rebuilds OULAD baselines with DT timeout and plateau stopping.
