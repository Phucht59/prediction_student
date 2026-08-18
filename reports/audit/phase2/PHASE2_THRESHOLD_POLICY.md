# Phase 2 — Threshold Policy

| Concept | Name | Objective | Fit data | Model selection? |
| --- | --- | --- | --- | --- |
| Diagnostic monitor | `monitor_threshold` | Fixed 0.5 reporting | None | No |
| Research evaluation | `research_threshold` | Maximize Macro-F1 | Pooled inner OOF only | No; applied after checkpoint selection |
| Operational intervention | `operational_threshold` | Maximize risk recall subject to precision ≥ 0.75 | Pooled inner OOF only | No |

The APIs accept explicitly named `inner_oof_labels` and
`inner_oof_probabilities`; outer labels are absent from their signatures.
Operational thresholds are retained but cannot choose epoch, architecture, or
scientific hyperparameters.
