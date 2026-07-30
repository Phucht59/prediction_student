# Phase 5 — Temporal Contribution

| Ablation | Macro-F1 |
| --- | --- |
| H1_FULL | 0.775677 |
| H1_WITH_RESIDUAL_TABULAR_LOGIT_DISABLED | 0.762289 |
| H1_WITH_TEMPORAL_BRANCH_DISABLED | 0.757738 |

Disabling temporal information changes Macro-F1 by `-0.017939`; disabling the residual changes it by `-0.013388`. Both pathways contribute materially. Temporal contribution: **STRONG**.
