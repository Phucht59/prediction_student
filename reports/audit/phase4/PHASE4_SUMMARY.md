# Phase 4 — Controlled Fusion Search

## Outcome

Gate: **PASS**. A1 vector gating was the numerical stability leader by only
`0.000442` Macro-F1, below the `0.002` materiality threshold. It also worsened
all preregistered secondary metrics versus A0. Concat+MLP and FiLM did not improve
the control in screening.

## Scientific conclusion

**D. FUSION/STAGE CONDITIONING DO NOT MATERIALLY HELP.**

Scalar gating was **not confirmed** as a material bottleneck. Explicit stage context
already exists, so duplicative stage conditioning was skipped. The current frozen
choice remains `A0_SCALAR_GATE`.

Should temporal CNN depth now be tested? **YES, BUT ONLY AS A CONTROLLED 1-vs-2
BLOCK ABLATION.** Pooling, fusion, training objective, threshold policy, and stage
policy should remain frozen in that later experiment.

## Stability evidence

| Architecture | Macro-F1 | Across-run SD | Worst | PR-AUC | NLL | Brier | ECE |
| --- | --- | --- | --- | --- | --- | --- | --- |
| A0_SCALAR_GATE | 0.773125 | 0.001274 | 0.705526 | 0.829182 | 0.448103 | 0.147006 | 0.018471 |
| A1_VECTOR_GATE | 0.773567 | 0.001103 | 0.704656 | 0.829169 | 0.448827 | 0.147231 | 0.021350 |
| A2_CONCAT_MLP | 0.771957 | 0.001757 | 0.703876 | 0.828090 | 0.450603 | 0.147778 | 0.021649 |

## Stage deltas: numerical winner versus control

| Stage | A0 | A1_VECTOR_GATE | Δ Macro-F1 | Δ PR-AUC | Δ NLL |
| --- | --- | --- | --- | --- | --- |
| 20% | 0.705526 | 0.704656 | -0.000870 | -0.000029 | 0.001677 |
| 35% | 0.744264 | 0.744452 | 0.000187 | -0.000408 | 0.001166 |
| 50% | 0.793391 | 0.794445 | 0.001054 | -0.000181 | 0.000311 |
| 75% | 0.849320 | 0.850716 | 0.001396 | 0.000571 | -0.000255 |
