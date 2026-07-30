# Phase 2 — Training Pipeline Repair

## Outcome

Phase 2 gate: **PASS**. Correctness repairs are implemented without modifying
official final metrics, reports, mappings, or checkpoints. The controlled
experiment is labelled `DIAGNOSTIC_ONLY` and used outer fold 0 only as the
held-out partition definition; all epoch and threshold choices used its two
inner train/validation splits, fixed seed 42, and no outer labels.

The four-epoch finding is **INCONCLUSIVE**. Mean validation NLL improved by
-0.0004 after epoch 4; mean fixed-threshold Macro-F1 changed by up
to +0.0007. The selected inner-fold NLL epochs were
[3, 9], deterministically aggregated to
6 for an eventual fixed refit.

The Phase 2 answer is: the CNN-BiLSTM **was limited by incorrect training
control and provenance**, but this diagnostic does **not** show that the
four-epoch budget itself materially suppressed performance. Correctness must
be repaired before architecture attribution, while strong tabular aggregates
and calibration remain plausible explanations for ML competitiveness.

## Boundaries

- Architecture topology/capacity was not changed; concat dimension correctness is the only model fix.
- Optuna was not executed.
- No outer metric was computed or used.
- No final experiment was rerun.
- CNN depth should **not** be changed in Phase 2.
