# Phase 4 — Stability

| Architecture | Macro-F1 | Across-run SD | Worst | PR-AUC | NLL | Brier | ECE |
| --- | --- | --- | --- | --- | --- | --- | --- |
| A0_SCALAR_GATE | 0.773125 | 0.001274 | 0.705526 | 0.829182 | 0.448103 | 0.147006 | 0.018471 |
| A1_VECTOR_GATE | 0.773567 | 0.001103 | 0.704656 | 0.829169 | 0.448827 | 0.147231 | 0.021350 |
| A2_CONCAT_MLP | 0.771957 | 0.001757 | 0.703876 | 0.828090 | 0.450603 | 0.147778 | 0.021649 |

A1 is the numerical Macro-F1 leader by 0.000442, but this is NEGLIGIBLE. It is worse than A0 on worst-stage Macro-F1, PR-AUC, NLL, Brier, and ECE. Therefore A0 is retained under the materiality and secondary-metric rule; no non-control fusion is a scientific winner.
