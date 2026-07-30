# Phase 5 — Screening

| Candidate | Macro-F1 | Worst | PR-AUC | NLL | Brier | ECE | Parameters |
| --- | --- | --- | --- | --- | --- | --- | --- |
| H0_CURRENT_HYBRID | 0.774889 | 0.708569 | 0.830746 | 0.447370 | 0.146707 | 0.022651 | 150202 |
| H1_TABULAR_RESIDUAL_EXPERT | 0.775972 | 0.710799 | 0.833522 | 0.444813 | 0.145743 | 0.024098 | 160492 |
| M0_MLP | 0.770910 | 0.702327 | 0.826928 | 0.459666 | 0.149118 | 0.033231 | 13569 |

H1-H0 = `0.001083`. The primary +0.002 trigger was not met, but the preregistered compensating PR-AUC/NLL trigger passed, so stability was run.
