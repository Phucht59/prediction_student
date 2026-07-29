# Phase 1 — Optuna Lineage

## Conclusion

**FINAL ARCHITECTURE IS NOT FULLY OPTUNA-TUNED.**

The retained OULAD Optuna evidence belongs to an earlier F2-only architecture
study, not to the current unified multi-kernel, gated-residual, multitask
factory.

## Historical OULAD study

| Field | Evidence |
| --- | --- |
| Run ID | `oulad-deep-v2-f2-20260716-v1` |
| Source branch | `feature/study-b-oulad-extension` |
| Source commit | `fccaef8b3e73a375f2a9d1bca2cc5897345242bd` |
| Protocol | `oulad_deep_v2_protocol_v1` |
| Forecast | `F2_MIDDLE` only |
| Outer / inner folds | 3 / 2 |
| Trials | 72, all `COMPLETE` |
| Search signal | Validation NLL |
| Trial ranking | Pooled inner-OOF Macro-F1 |
| Pruning | None recorded |
| Candidates | V2-MLF, V2-H2F, V2-H2T, V2-A0, V2-T0, V2-H3C |

The H2T search tuned a single kernel (`3` or `5`), convolution channels, LSTM
hidden/layers, dropout, LR, WD, batch size, scheduler, and positive-weight
policy. H3C reused selected temporal and aggregate configs with concatenation.

It did **not** tune:

- the current parallel kernels `[2, 3, 5]`;
- current `gated_residual` fusion;
- the current masked mean+max pooling choice;
- survival/outcome multitask weights;
- branch dropout;
- the current `CNNBiLSTMOULAD` architecture factory;
- the unified one-checkpoint/four-stage objective.

## Unified OULAD

`configs/final/oulad_prediction.yaml` contains exactly one deep config:
`frozen_default`. `inner_trials.csv` confirms that all unified deep families
use this ID. There is no unified Optuna study; max epochs are 4 and patience is
2.

## UCI distinction

The official V5.1 Student-Mat and Student-Por models do have broader historical
Optuna/component evidence, including fusion, objective, imbalance, LR, WD,
dropout, depth, and hidden dimensions. That lineage must not be transferred to
the unified UCI experiment, whose deep candidate grid contains only two configs
per family/fold and forces one LSTM layer, one context layer, GELU, and gated
fusion.

Unified UCI is therefore classified **LIMITED DEEP SEARCH**, not comprehensive
hyperparameter optimization.

Machine-readable lineage is in
`artifacts/audit/phase1/optuna_lineage.json`.
