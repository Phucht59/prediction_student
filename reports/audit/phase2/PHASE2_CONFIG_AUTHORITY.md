# Phase 2 — Configuration Authority

The single authority for corrected unified stage-aware OULAD development is
`configs/registry/oulad_unified_stage_aware_v2.yaml`. The legacy official
single-cutoff config remains separate and frozen.

| Scope | Config | Parameters | Pretraining executed |
| --- | --- | ---: | --- |
| Legacy official frozen | `configs/final/cnn_bilstm_oulad.yaml` | 100,938 | Historical config declares a strategy; execution is not inferred |
| Unified stage-aware v2 | `configs/registry/oulad_unified_stage_aware_v2.yaml` | 150,202 | No |

The 49,264-parameter difference is not
hidden: the unified runtime fingerprints actual sequence (47), aggregate
(165), static (13), representation (64), heads, architecture, loss,
pretraining, and parameter count. `config_hash` covers configuration;
`architecture_hash` additionally binds runtime dimensions and model class.
