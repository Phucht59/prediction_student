# Nested cross-validation audit — Scientific Protocol V2

## Evidence inspected

- Selection run `artifacts/model_selection/nested-full-20260710` declares 5 stratified outer folds, 3 inner folds, 30 Optuna trials, seed 42, and `locked_test_used_for_selection: false`.
- Historical final run `a2945d79-9845-4979-b148-159f4853eca3` used the 316/79 fixed membership; PostgreSQL verification run `5a0b5041-5216-4a48-9e46-b0c16ab14866` reproduced the 79 predictions.
- Relevant implementation: `scripts/optimize_model_selection.py`, `src/model_selection.py`, `src/train_pipeline.py`, `src/data_pipeline.py`, and migrations `001`–`003`.

## Actual protocol before V2 refit correction

```text
316 historical development records
  -> StratifiedKFold(5) outer train / outer validation
  -> outer train: Optuna 30 trials over StratifiedKFold(3) inner scoring folds
  -> each fold train: 85% model-train + 15% internal early-stop
  -> preprocessing/selector/resampling fit on model-train only
  -> early stopping + LR scheduler observe internal early-stop only
  -> outer validation: inference once
```

The current implementation did **not** pass outer-validation to `train_model`, the scheduler, class weights, preprocessing fit, feature selection, threshold selection, or calibration. This is evidenced by `src/model_selection.py:fit_fold_predict_proba`, which passes `early_stop_loader` to `train_model` and uses `validation_loader` only after training. The fixed 79 rows are reconstructed only by legacy paths and were not passed to the historical Optuna objective.

However, the pre-V2 implementation evaluated each outer fold with a model trained on only the 85% model-train subset. It selected an epoch internally but did not refit on the full outer-train partition. The legacy final inference path similarly held 15% of the 316 records for early stopping, so the saved historical model was trained on approximately 268 records, not a full-316 refit. This is a protocol limitation, not a changed historical result.

## V2 correction

`src/model_selection.fit_fold_predict_proba` now performs:

```text
outer train
  -> stratified internal model-train / early-stop split
  -> choose epoch solely from early-stop F1 (scheduler also sees only this split)
  -> refit scaler/encoder/selector/resampler on 100% outer train
  -> fixed-epoch model fit on 100% outer train
outer validation -> one inference/evaluation only
```

`src.train_pipeline.train_fixed_epochs` has no validation loader and no scheduler. Thus the scoring partition cannot influence epoch, checkpoint, learning rate, threshold, calibration, hyperparameters, feature selection, preprocessing, resampling, or class weights. The resulting model-selection code also loads the immutable V2 outer-fold manifest and rejects the observed legacy-79 IDs.

## Remaining risks and boundaries

- The 79 rows were historically examined by several analyses; they are permanently `legacy_heldout_observed`, never evidence for future selection.
- The 316 development cohort has also been used historically; V2 therefore reports nested/repeated nested CV uncertainty, not a newly “locked” test claim.
- A serialized historical checkpoint is absent (`model_checksums.json` has an empty checkpoint map); the legacy manifest records this explicitly.
- Repeated nested CV and multi-seed stability are supported as future protocol work, but are intentionally not run in this cleanup phase.
