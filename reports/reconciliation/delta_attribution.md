# Delta attribution

Historical fold mean is 0.8781; valid V2 five-seed estimator is 0.7984 (delta −0.0797). The terms interact and are not asserted to add exactly.

| Component | Evidence-based estimated effect |
|---|---:|
| Fold manifest | 0.0000: exact same 316 records and 316 fold assignments |
| V2 clean refit protocol | −0.0166: A 0.8781 → C 0.8615 using identical historical per-fold configs and seed 42 |
| Configuration/search-space/training-budget drift | about −0.0361 on seed 42: C 0.8615 → E 0.8254; includes 20 vs 40/60 epochs, lower LR range, no focal candidate, no SMOTE candidate, and changed capacity/kernel options |
| Seed aggregation/variation | −0.0270 between V2 seed-42 0.8254 and five-seed fold estimator 0.7984; this is an estimator difference, not an independent additive causal effect |
| Metric aggregation | negligible for historical claim: fold mean 0.878089 versus pooled OOF 0.877926 |
| Implementation drift not isolated | residual/confounded; historical checkpoints and epoch histories are unavailable |

Primary conclusion: the observed delta is a combination of clean-refit change, a materially different tuning/training search space, and multi-seed aggregation. It is not explained by fold assignment and no outer-validation training leakage was found in the historical code revision.
