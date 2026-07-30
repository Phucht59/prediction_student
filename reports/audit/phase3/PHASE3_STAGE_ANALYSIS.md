# Phase 3 — Stage Analysis

| Stage | Control F1 | Selected F1 | ΔF1 | ΔPR-AUC | ΔNLL |
| --- | ---: | ---: | ---: | ---: | ---: |
| E1_EARLY_20PCT | 0.701910 | 0.709131 | +0.007221 | +0.010117 | -0.013552 |
| E2_EARLY_35PCT | 0.739413 | 0.747648 | +0.008235 | +0.007785 | -0.011607 |
| M1_MIDDLE_FROZEN | 0.790650 | 0.795323 | +0.004673 | +0.004399 | -0.008285 |
| L1_LATE_75PCT | 0.850243 | 0.849700 | -0.000542 | +0.001866 | -0.005267 |

Tuning helps early/middle Macro-F1 most. The 75% stage is approximately flat,
which reinforces using equal-stage aggregation instead of optimizing only the
late stage.
