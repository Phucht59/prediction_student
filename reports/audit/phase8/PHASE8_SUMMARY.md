# Phase 8 — OULAD Endpoint Forensic Audit

## Executive conclusion

The endpoint regression is real on the same 15,378 records, targets, cutoff
days and outer folds: H0 reproduces at **0.828084**, H1 at
**0.798400**, delta **-0.029684**.

The root-cause classification is **I — MULTIPLE FACTORS**. Phase 7 did not
evaluate “historical H0 plus a residual expert.” It evaluated the Phase 5
early-warning H1 recipe at the F2 endpoint. Compared with H0, it changed the
feature authority (most importantly endpoint score-progress), temporal
topology, preprocessing, pretraining, inner-fold count, training
hyperparameters, auxiliary weights and epoch policy.

This is not a threshold-only problem. H1 also loses PR-AUC
(0.863039 vs 0.893355), ROC-AUC
(0.876142 vs 0.908156) and NLL
(0.406292 vs 0.358778). A diagnostic outer-oracle threshold
would improve H1 by only
+0.000707
and is not used for selection.

## Decision

Recovery path: **R1 — H0 protocol components were better and valid under the
historical endpoint contract**. A future H1-R study may combine H1 with a
scientifically re-authorized score-availability contract, H0 train-only
preprocessing and H0 endpoint pretraining/training recipe. No such model is
trained in Phase 8. Because Phase 7 outer labels are already known, any
corrected endpoint candidate requires a genuinely new untouched holdout.

H1 remains frozen and recommended for early-warning. H0 remains the endpoint
authority until a new development-only recovery study and new holdout exist.
