# V6 knowledge audit

All selection-facing analyses below use only outer-training fold 0 and its
cross-fitted inner folds. Future OULAD remained locked and no outer-test result
was used for a gate.

## Temporal order destruction

| Variant | Macro-F1 | At-risk F1 | PR-AUC | Brier | Δ Macro-F1 |
|---|---:|---:|---:|---:|---:|
| bag_of_weeks | 0.818729 | 0.778085 | 0.885137 | 0.119383 | -0.002526 |
| original | 0.821255 | 0.777361 | 0.888963 | 0.117389 | +0.000000 |
| reversed | 0.821807 | 0.779735 | 0.889324 | 0.117082 | +0.000552 |
| shuffled | 0.820468 | 0.778709 | 0.885611 | 0.119367 | -0.000787 |

Verdict: **TEMPORAL_ORDER_LOW_VALUE**. Every variant used the same V5.1 architecture,
seed, eight-epoch budget and cross-fitted threshold protocol.

## Residual ceiling

The diagnostic classifier used the 64-dimensional cross-fitted V5.1 temporal
projection to predict whether XGBoost was correct. Residual AUC was
`0.668940` and residual PR-AUC was
`0.908460`. Verdict:
**RESIDUAL_SIGNAL_HIGH**. Complex selector allowed:
`true`.

## Oracle complementarity

- Deep correct / XGBoost wrong: 275
- XGBoost correct / Deep wrong: 324
- Both correct: 8276
- Both wrong: 1383
- Disagreement rate: 0.058393
- Oracle-union accuracy: 0.865178
- Oracle gain over best: +0.026808

The oracle is diagnostic only and is not a deployable selector.

## Survival feasibility

Valid withdrawal timestamps exist for 1160
historical records; 1156 events occur
after the F2 cutoff. Fail is not treated as a withdrawal event. Verdict:
**WITHDRAWAL_SURVIVAL_FEASIBLE**.

## Graph context

The audit found 4 graph-only descriptors
with within-module presentation variation. Verdict: **GRAPH_CONTEXT_PASS**.
This gate only permits a small context embedding; it does not select one.
