# Phase 3 — Efficient Optuna VNext

## Outcome

Gate: **PASS**. All 72 scheduled trials completed or were validly pruned:
46 COMPLETE, 26 PRUNED, 0 FAILED, and 0 OOM. Architecture hash count and
parameter-count count are both one; pretraining and outer-label access remained
disabled.

At the search seed, tuned configurations improved mean-stage Macro-F1 by
+0.004897 (SMALL). Across the two
preregistered stability seeds and three folds, the mean delta was only
+0.001736 (NEGLIGIBLE), with
positive direction in 4/6 fold-seed pairs.

## Final classification

**C. CURRENT ARCHITECTURE IS NEAR ITS TRAINING OPTIMUM.**

Training hyperparameter tuning improves NLL/Brier and usually Macro-F1, but the
stability Macro-F1 gain is negligible under the project materiality rule. The
current architecture is therefore near its training optimum rather than
materially under-tuned.

Should CNN be deepened now? **NOT JUSTIFIED; PRIORITIZE OTHER ARCHITECTURAL
HYPOTHESES.**
