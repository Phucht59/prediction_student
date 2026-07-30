# Phase 3 — Control vs Tuned

## Search-seed inner evidence

| Metric | Control | Selected | Delta (selected-control) |
| --- | ---: | ---: | ---: |
| mean_stage_macro_f1 | 0.770554 | 0.775451 | +0.004897 |
| worst_stage_macro_f1 | 0.701910 | 0.709131 | +0.007221 |
| mean_stage_pr_auc | 0.824603 | 0.830645 | +0.006042 |
| mean_stage_nll | 0.457191 | 0.447513 | -0.009678 |
| mean_stage_brier | 0.149848 | 0.146630 | -0.003218 |
| mean_stage_ece | 0.030302 | 0.022918 | -0.007385 |

Macro-F1 materiality: **SMALL**. NLL, Brier and ECE
are lower for selected configurations. These are development inner metrics,
not outer final results.
