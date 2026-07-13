# Scientific Protocol V2

## Scope and legacy boundary

The 79-record split is not a fresh test set. Its predictions were inspected for CNN–BiLSTM, G2 rule, Logistic Regression, HistGradientBoosting, ablations, resampling/class weights, and multi-seed ensembles. It is preserved as `legacy_heldout_observed` for historical reconciliation only. It must never select architecture, feature set, loss, seed, ensemble, threshold, resampling, or hyperparameters, and it cannot decide whether a future model is better.

The evaluable cohort is the existing 316-record development membership recorded in `artifacts/protocol_v2/student_mat_development_outer_folds.json`. This is not a newly fabricated locked test; absent wholly new external data, all future claims are nested-CV claims on this cohort.

## Required procedure

1. Use the shared deterministic 5-fold stratified outer manifest (seed 42). Every candidate—including rules, linear, tree, neural, ordinal, and hybrid methods—must use these exact outer validation records.
2. For each outer fold, tune only inside outer-train using 3-fold stratified inner CV and 30 Optuna trials (or a pre-registered equivalent). The main objective is macro-F1.
3. Choose epoch using an internal stratified split of outer-train. Fit scaler, encoder, imputer, selector, and resampler only on the applicable training partition. Then refit all such transforms and the model on all outer-train with that fixed epoch. Outer-validation is inference/evaluation exactly once.
4. SMOTE/ADASYN or class weighting are train-only choices. No scoring fold can enter resampling, class-weight computation, early stopping, scheduler stepping, checkpoint selection, thresholding, calibration, or feature selection.
5. Feature inputs must pass the strict scenario allowlist. Pre-assessment currently has no admissible UCI variables; early-warning permits G1; late-stage permits G1/G2. Unknown availability requires an approved data-contract change, not an ad-hoc override. G3 and G3-derived fields are always prohibited.

## Evaluation and comparison

Each fold emits record-level predictions with model/scenario/feature-set/seed/trial/config checksum/fold checksum/dataset checksum/code commit. Primary metric: macro-F1. Secondary metrics: accuracy, weighted F1, balanced accuracy, QWK, ordinal MAE, one-/two-step errors, confusion matrix, and Brier/PR-AUC when probabilities exist.

For future robustness work, pre-register multiple seeds and optionally repeated nested CV. Compare candidates paired on exactly the same outer-validation rows; report fold-wise differences, mean/SD, paired bootstrap confidence intervals, class-wise changes, and ordinal error changes. A candidate is better only when its pre-specified primary metric improves with a compatible confidence interval and no material regression in class coverage/ordinal harm; a one-off observed-79 score never qualifies.

Run paired comparison after two V2 prediction artifacts exist:

```powershell
py -3.10 -m src.evaluation.compare_runs --run-a <record_predictions_a.csv> --run-b <record_predictions_b.csv> --metric macro_f1
```

Run future selection only through the shared manifest:

```powershell
py -3.10 scripts/optimize_model_selection.py --dataset student-mat --dataset-version-id 1 --fold-manifest artifacts/protocol_v2/student_mat_development_outer_folds.json --outer-folds 5 --inner-folds 3 --n-trials 30
```

External evaluation is permitted only with a genuinely newly collected external dataset/version, frozen before model choice. Until then, report nested-CV uncertainty and do not simulate a new locked test by randomly re-splitting these 395 observed records.
