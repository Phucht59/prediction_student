# V6 final prediction report

Selected candidate: **C_TEMPORAL_MULTITASK_W0**. Selection was frozen from
inner-fold gates before viewing any outer-test result.

- Outer evaluation: 3 folds x 5 fixed seeds (15 checkpoints)
- Ensemble Macro-F1: 0.828084
- Ensemble At-risk F1: 0.782639
- Ensemble PR-AUC: 0.893355
- Ensemble Brier: 0.113355
- Ensemble ECE: 0.008683
- Recall@10%: 0.250409
- Survival C-index: 0.641216
- Outcome Macro-F1: 0.615295
- Parameters: 100,938
- Total recorded fit runtime: 3080.5 seconds
- Peak CUDA allocation: 125.2 MiB

XGBoost remains an operational cross-check and is not embedded in the deep
model. Future OULAD remained locked.
