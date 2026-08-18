# OULAD 75% versus FINAL audit

The old comparison was not 75% versus FINAL: Phase 7's endpoint was F2 at 50%
and used a separate checkpoint. Score policy was the same strict score-free
policy. Classification: **G_OLD_STAGE_EVIDENCE_NOT_PROTOCOL_COMPATIBLE, C_SEPARATE_TRAINING_MISMATCH**.

## Canonical same-checkpoint diagnostic

| stage | macro_f1 | pr_auc | roc_auc | nll | brier | ece |
|---|---|---|---|---|---|---|
| L1_LATE_75PCT | 0.852491 | 0.906029 | 0.919790 | 0.311731 | 0.095938 | 0.005896 |
| FINAL | 0.894556 | 0.933686 | 0.943434 | 0.245062 | 0.072175 | 0.009050 |

FINAL − 75%: Macro-F1 0.042065, PR-AUC
0.027658, ROC-AUC 0.023644, NLL
-0.066669.

## Same-checkpoint common-cohort diagnostic

The common cohort contains **14243** records, removing dynamic
risk-set composition as an explanation for the information-time comparison.

| stage | macro_f1 | pr_auc | roc_auc | nll | brier | ece |
|---|---|---|---|---|---|---|
| L1_LATE_75PCT | 0.856493 | 0.903884 | 0.921583 | 0.301162 | 0.091842 | 0.003731 |
| FINAL | 0.894556 | 0.933686 | 0.943434 | 0.245062 | 0.072175 | 0.009050 |

Common-cohort FINAL − 75%: Macro-F1 0.038062,
PR-AUC 0.029803, ROC-AUC
0.021851, NLL -0.056100.

## Canonical task-specific checkpoints

The primary benchmark separately reports the shared-stage checkpoint at 75%
and the dedicated FINAL-trained checkpoint. FINAL − 75% is Macro-F1
0.041580, PR-AUC 0.028959,
ROC-AUC 0.025174, NLL -0.069443.
