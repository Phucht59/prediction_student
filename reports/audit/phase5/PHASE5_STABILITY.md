# Phase 5 — Stability

| Candidate | Macro-F1 | SD | Worst | PR-AUC | NLL | Brier | ECE |
| --- | --- | --- | --- | --- | --- | --- | --- |
| H0_CURRENT_HYBRID | 0.773598 | 0.000953 | 0.705529 | 0.829329 | 0.448292 | 0.147029 | 0.020064 |
| H1_TABULAR_RESIDUAL_EXPERT | 0.775677 | 0.001894 | 0.709110 | 0.831940 | 0.445882 | 0.146167 | 0.022039 |
| M0_MLP | 0.769597 | 0.003730 | 0.700617 | 0.823190 | 0.467519 | 0.151049 | 0.039439 |

H1 exceeds H0 in 6/6 pairs and MLP in 6/6 pairs. H1-H0 = `0.002079`; H1-M0 = `0.006080`. The MLP comparison meets the preregistered strong-inner-win definition.
