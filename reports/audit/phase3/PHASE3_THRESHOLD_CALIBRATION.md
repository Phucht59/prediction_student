# Phase 3 — Threshold and Calibration

Mean research-threshold range across folds changed from
0.123682 to 0.077312
(-0.046370): **Improved**.

| Stability metric | Control | Selected | Delta |
| --- | ---: | ---: | ---: |
| mean_stage_nll | 0.457429 | 0.448194 | -0.009235 |
| mean_stage_brier | 0.150289 | 0.146973 | -0.003316 |
| mean_stage_ece | 0.039531 | 0.020166 | -0.019365 |

Calibration classification: **Improved**. No Platt, isotonic, or
temperature scaling was introduced. Thresholds remain stage-specific and
inner-only.
