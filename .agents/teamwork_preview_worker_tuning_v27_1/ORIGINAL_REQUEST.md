## 2026-06-15T15:20:02+07:00
Please perform the following tasks:
1. Create `scripts/run_v27_optuna.py`:
   - It must implement an Optuna study with 50 trials for each of the three datasets: `student-mat`, `student-por`, and `xapi`.
   - For each trial, run 5-fold cross-validation on the training pool. Inside the fold: split, engineer features, apply feature selection on train subset, oversample with SMOTENC/SMOTE (never on validation), build `StudentHybridV27` and train it using `JointHybridLoss` and Adam optimizer.
   - The metric to maximize is the average F1-Macro across validation folds.
   - Save the best parameters found to `models/saved/final/{dataset}_3class_best_params.json`.
   - Make sure you support `--dataset` argument and `--n-trials` argument.

2. Create `scripts/tune_v27_thresholds.py`:
   - For each of the three datasets, load the best parameters from `models/saved/final/{dataset}_3class_best_params.json`.
   - Train the `StudentHybridV27` model using 5-fold cross-validation.
   - Collect the out-of-fold validation class probabilities.
   - Search for a probability threshold modifier or class multiplier for class 0 (Low) that maximizes a combined validation objective: `0.5 * F1_Macro + 0.5 * Recall_Low`.
   - Save the tuned thresholds to `outputs/experiments/thresholds_{dataset}.json`. The file format must be:
     `{"threshold_low": <float>, "class_multipliers": [<float>, <float>, <float>]}`.

3. Run the scripts:
   - Run the Optuna tuning script for `student-mat`, `student-por`, and `xapi` (you can use `--n-trials 50` or a lower count if time is constrained, but try to run 50 trials as requested).
   - Run the threshold tuning script for all three datasets to generate the threshold JSON files.
   - Run `scripts/run_v27_pipeline.py` for each dataset to verify that the final cross-validation pipeline runs successfully and utilizes the newly saved best parameters.

Write a complete report of the tuning results, best hyperparameters, validation metrics before and after threshold tuning, and verify that the output files exist. Write the report to `c:\Huflit\kltn\.agents\teamwork_preview_worker_tuning_v27_1\handoff.md` and send a message when complete.
