# Phase 1 — Baseline Freeze

## Frozen official results

| Model | Dataset | Stage/protocol | Macro-F1 | PR-AUC | NLL | Brier | ECE | Authority |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| CNN-BiLSTM | Student-Mat | Official final, G1+G2 | 0.901460 | 0.944184 | 0.363513 | — | — | `artifacts/final/metrics/cnn_bilstm_mat.json` |
| CNN-BiLSTM | Student-Por | Official final, G1+G2 | 0.862259 | 0.914679 | 0.307893 | — | — | `artifacts/final/metrics/cnn_bilstm_por.json` |
| CNN-BiLSTM | OULAD | Official F2 middle single-cutoff | 0.828084 | 0.893355 | 0.358778 | 0.113355 | 0.008683 | `artifacts/final/metrics/cnn_bilstm_oulad.json` |

OULAD `0.828084` is **not** a 100% stage result. It is the frozen official
`F2_MIDDLE` result under the earlier single-cutoff protocol, using per-outer-fold
thresholds `0.455`, `0.495`, and `0.500`.

## Frozen unified OULAD stage-aware results

These are a different authority and checkpoint family. Values below use the
registered `INNER_OOF_STAGE_THRESHOLD` operational policy.

| Stage | Mean threshold | Macro-F1 | Macro precision | Macro recall | PR-AUC | Brier | NLL | ECE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 20% | 0.687084 | 0.700323 | 0.724302 | 0.699158 | 0.762423 | 0.204659 | 0.592877 | 0.127438 |
| 35% | 0.606186 | 0.743506 | 0.757862 | 0.739772 | 0.808627 | 0.173336 | 0.516599 | 0.097896 |
| 50% | 0.446797 | 0.785192 | 0.783951 | 0.787725 | 0.859072 | 0.137354 | 0.424336 | 0.061745 |
| 75% | 0.310894 | 0.806211 | 0.802794 | 0.821260 | 0.903516 | 0.100588 | 0.326819 | 0.047840 |

The mean thresholds are descriptive means over the three fold-specific
inner-OOF thresholds. Evaluation applies each fold's own threshold, not this
mean.

## Freeze policy

- Official and unified evidence was read, not recomputed or overwritten.
- Fold-level calibration and threshold diagnostics added by Phase 1 are marked
  `RECOMPUTED_FOR_AUDIT_FROM_FROZEN_PREDICTIONS`.
- Diagnostic values are not promoted into official result tables.
- No outer label was used to choose a model, checkpoint, threshold, or fix.
- Complete machine-readable rows, including fold/seed/stage/checkpoint fields,
  are in `artifacts/audit/phase1/baseline_metrics.json`.
