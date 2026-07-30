# Phase 4 — Fusion Screening

| Architecture | Macro-F1 | Worst | PR-AUC | NLL | Brier | ECE | Parameters |
| --- | --- | --- | --- | --- | --- | --- | --- |
| A0_SCALAR_GATE | 0.774756 | 0.708116 | 0.830164 | 0.447543 | 0.146651 | 0.020980 | 150202 |
| A1_VECTOR_GATE | 0.774011 | 0.708529 | 0.829558 | 0.448935 | 0.147239 | 0.023082 | 155080 |
| A2_CONCAT_MLP | 0.774018 | 0.704927 | 0.828718 | 0.449484 | 0.147131 | 0.019197 | 162600 |
| A3_FILM | 0.773676 | 0.707065 | 0.829807 | 0.449721 | 0.147618 | 0.026859 | 154056 |

A1 and A2 advanced as the two highest-ranked non-control candidates. No Stage A candidate improved control Macro-F1.
