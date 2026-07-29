# Phase 1 — Metric Authority Map

## Authorities

| Evidence type | Source of truth | Notes |
| --- | --- | --- |
| Official final comparator | `artifacts/final/final_results.json` and `.csv` | Frozen single official result per registered model |
| Official model detail | `artifacts/final/metrics/cnn_bilstm_{mat,por,oulad}.json` | Seed, per-class, calibration/ranking details where available |
| Unified stage-aware UCI | `artifacts/final/unified_stage_aware_uci/stage_metrics.csv` | S0/S1/S2, one estimator across stages |
| Unified UCI overall | `artifacts/final/unified_stage_aware_uci/overall_metrics.csv` | Stage aggregate, not official final replacement |
| Unified stage-aware OULAD | `artifacts/final/unified_stage_aware_oulad/stage_metrics.csv` with `INNER_OOF_STAGE_THRESHOLD` | 20/35/50/75 operational authority |
| Unified OULAD overall | `artifacts/final/unified_stage_aware_oulad/overall_metrics.csv` | Four-stage aggregate |
| Project unified mirror | `artifacts/final/final_stage_results.csv`, `final_overall_results.csv` | Consolidated mirror; trace to dataset-specific authority |
| OULAD threshold policy | `artifacts/final/unified_stage_aware_oulad/threshold_policies.csv` | Fold/stage inner-OOF thresholds |
| Calibration | Official calibration JSON for frozen official; unified `stage_metrics.csv`; Phase 1 fold detail in `calibration_audit.csv` | Phase 1 detail is recomputed for audit |
| Ablation | `artifacts/final/ablation_evidence/*.csv` and `final_report.json` | Controlled development diagnostic, not final outer result |
| Tuning | `artifacts/final/tuning_evidence/<model>/` | Historical protocol-specific evidence |
| Bootstrap CI | Dataset-specific `bootstrap_stage.csv`, `bootstrap_overall.csv`, and `artifacts/final/bootstrap/` | Keep protocol and resampling unit attached |

## Reconciled collisions

### OULAD Macro-F1 near 0.828

- `0.828084`: official CNN-BiLSTM, F2 single-cutoff, five-seed ensemble.
- `0.827422`: reused historical architecture-diagnosis reference under a
  development diagnostic protocol.
- `0.785192`: unified four-stage estimator at M1/50% with operational
  inner-OOF threshold.
- `0.793406`: the same unified frozen probabilities at M1 with fixed threshold
  0.5.

These are not interchangeable. None is stage 100%.

### Unified OULAD threshold policies

`FIXED_0_5` and `INNER_OOF_STAGE_THRESHOLD` share identical probabilities,
PR-AUC, ROC-AUC, Brier, NLL, and ECE but differ in thresholded metrics. The
registered unified operational authority is the inner-OOF policy. Fixed 0.5 is
a protocol comparator, not an outer-selected replacement.

### Official versus unified UCI

Official UCI values (`0.901460` MAT, `0.862259` POR) come from V5.1 final
ensembles and should not be replaced by unified S2 values. The unified study
uses a deliberately limited two-config, shared-stage training protocol.

## Selection rule

When duplicate-looking numbers exist, choose by protocol, stage, fold, seed,
checkpoint, and evidence version—not by the larger value. Phase 1 did not
rewrite any final report.
