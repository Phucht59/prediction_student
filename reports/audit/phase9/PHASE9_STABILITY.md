# Phase 9 stability

Two predefined seeds were evaluated across each of three development
outer-train partitions.

| Candidate | Macro-F1 mean | std | PR-AUC | ROC-AUC | NLL | Brier | ECE |
|---|---:|---:|---:|---:|---:|---:|---:|
| H1-R0 Phase 7 control | 0.796508 | 0.001153 | 0.858145 | 0.872221 | 0.412673 | 0.132927 | 0.014683 |
| Selected H1_R0_PHASE7_CONTROL | 0.796508 | 0.001153 | 0.858145 | 0.872221 | 0.412673 | 0.132927 | 0.014683 |

Recovery classification: **FAILED_RECOVERY**.

Historical development context (not used for selection) places score-proxy
H0 at 0.824443
and tabular MLP at 0.829920 on
outer-training-fold-0 inner evidence. This comparison is only partial because
those historical candidates used the score proxy rejected by Phase 9.
