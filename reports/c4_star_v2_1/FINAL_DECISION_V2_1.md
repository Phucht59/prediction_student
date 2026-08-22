DEVELOPMENT_GATE_FAILED

# FINAL_DECISION_V2_1

Outer test **not opened**. Serving Hybrid **not** promoted.

| Field | Value |
|---|---|
| Time | 2026-08-22T02:46:36Z |
| Commit | `ae883396b15294a075aecf47cd8c70998cd213f1` |
| Protocol | `c4_star_v2.1` |
| Hash | `ce758268ce0c834624a76f847864e4f31f553d85d1bb6458d453d07b8f8ee9ac` |

## Verified ceilings

- UCI CatBoost 3×3: S0 0.5010 / S1 0.7694 / S2 0.9067
- OULAD v2.1 3×3: 20% LR 0.7678; 35–100% XGB 0.8077 / 0.8545 / 0.8969 / 0.9245

## C4-STAR vs ceiling (official = robust 3×3, not HPO fold 0)

UCI M4 9-run mean: S0 0.493 / S1 0.775 / S2 0.856 vs CatBoost 0.501 / 0.769 / 0.907. S1 slightly positive but **not material**; S2 loses ~0.051. **UCI gate fail.**

OULAD M4 9-run mean loses every warm stage by ~0.002–0.005 vs XGB (35% 0.803 vs 0.808; 100% 0.921 vs 0.924). Cold 20% within guardrail. Fold-0 HPO had n_warm_loss=0; **3×3 does not replicate that.** **OULAD gate fail.**

Joint development gate requires both domains. Combined: **FAIL**. See `07_ROBUST_3X3_RESULTS.md`.

## Temporal

OULAD shuffle/reverse gaps are positive. Sequence order is real, not sufficient for material AP.

## Claims

Forbidden: vượt trội; OULAD 100% early-warning; SPEED_FINISH as confirmation.

Allowed: protocol v2.1 frozen; OULAD ceiling rebuilt 3×3; C0-R still best existing backbone on OULAD screen; C4-STAR M4 can match/slightly exceed the OULAD envelope on one fold without material margin; UCI Hybrid/C4 does not beat CatBoost at S2.
