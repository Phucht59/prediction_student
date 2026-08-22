# Hybrid lock — C0-R

Research prediction architecture is **C0-R** (parallel CNN ∥ BiLSTM, 3-way softmax). Frozen `2026-08-22T05:07:02Z`.

Serving Hybrid (Phase4 C0) is **unchanged**. `reports/CURRENT_REPORTS.md` is **untouched**. Outer test: **false**.

Defense status: **NOT_READY_FOR_DEFENSE** (no vượt trội claim). This lock freezes the Hybrid, not a serving cutover.

## Science checked

| Item | Status |
|---|---|
| AP primary, no G3, UCI G1/G2 temporal-only | pass |
| One checkpoint scores all stages | pass |
| Group-safe FIT/STOP/VALID, outer firewall | pass |
| Integrity tests | pass |
| UCI 3×3 authority | pass (S0 0.461 / S1 0.811 / S2 0.913) |
| OULAD 3×3 authority | pass (20 0.748 / 35 0.806 / 50 0.852 / 75 0.893 / 100 0.919) |
| Independent ablation 3×3 | **not** lock evidence (SPEED 8-epoch only) |
| Fair baseline | one weight per family; Optuna-best on stacked warm AP; protocol 40/28 trials |

## Recommendation

PASS — no rec code change

Rec reads `PredictionResult` (`model_id='hybrid'`). It does not inspect C0 vs C0-R weights. V3 refuses `100pct` as an intervention state. No rec file was edited.

## Chosen prediction

- Research: C0-R (`experiments/hybrid_superiority_v2`)
- Serving: Phase4 C0 (`src/prediction`, `configs/prediction/hybrid_final.json`)
