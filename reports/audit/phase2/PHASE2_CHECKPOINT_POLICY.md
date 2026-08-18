# Phase 2 — Checkpoint Policy

| Signal | Median inner-selected epoch |
| --- | ---: |
| fixed_macro_f1_at_0_5 | 3 |
| validation_nll | 6 |
| validation_pr_auc | 8 |
| threshold_optimized_macro_f1 | 8 |

## RECOMMENDATION

Use mean-stage validation NLL (minimize) to select a checkpoint independently
inside each inner fold, then propagate the round-half-up median epoch to the
outer full-training refit. Fit the research threshold afterward on pooled
inner OOF probabilities.

## RATIONALE

NLL is threshold-independent, reflects probability quality under the observed
calibration drift, and supports one shared checkpoint across four stages. It
does not couple checkpoint selection to the operational intervention objective.

## RISKS

NLL can favor calibration over the final Macro-F1 ranking. Two inner folds on
one preregistered outer partition are enough for controlled diagnosis but not
for claiming a final performance improvement.

## ALTERNATIVES

PR-AUC is the preferred preregistered sensitivity objective for Phase 3.
F1@0.5 is threshold-dependent. Inner-threshold Macro-F1 has nested-selection
optimism unless threshold fitting is nested again.
