## 2026-06-15T08:40:59Z
Your working directory is: c:\Huflit\kltn\.agents\teamwork_preview_worker_ensembling_v27_1
Your role is: Ensembling and Model Ablation analyst

Please perform the following tasks:
1. Create `scripts/run_v27_ensemble.py`:
   - It must implement seed ensembling over seeds 42, 43, 44, 45, and 46.
   - For each dataset (`student-mat`, `student-por`, `xapi`), load the best hyperparameters from `models/saved/final/{dataset}_3class_best_params.json`.
   - Train 5 separate `StudentHybridV27` models on the full training pool (using the best hyperparameters) with random seeds 42, 43, 44, 45, 46.
   - For each model, perform feature engineering, apply feature selection, fit data preprocessor on training pool (with SMOTENC/SMOTE oversampling), and train the model.
   - Load the tuned decision thresholds/class multipliers from `outputs/experiments/thresholds_{dataset}.json`.
   - Perform inference on the locked test set: average the class probabilities predicted by the 5 models. Apply the tuned decision thresholds/multipliers to the averaged probabilities.
   - Evaluate the metrics on the locked test set: Accuracy, Precision/Recall macro, F1-Macro, F1 for each of the 3 classes, and Recall for the Low group. If the dataset has a numeric grade (student-mat, student-por), also evaluate the regression head predictions on locked test using RMSE and R^2.
   - Save the results to `outputs/v27/{dataset}/ensemble_metrics.json`.

2. Create `scripts/run_v27_ablation.py`:
   - It must implement an ablation study containing exactly 10 variants of the V27 model, evaluated on the `student-mat` dataset.
   - The 10 variants to evaluate:
     1. Full V27 Model (Base).
     2. Context-only (no sequence branch, meaning seq_x is not used or masked).
     3. Sequence-only (no context branch, meaning num_x and cat_x are not used).
     4. Concatenation fusion (replace Gated Fusion with simple vector concatenation).
     5. No Attention Pooling (replace AttentionPooling1D with standard mean pooling on BiLSTM output).
     6. No Ordinal Auxiliary Head (set ordinal loss weight w_ord = 0 in JointHybridLoss).
     7. No Regression Auxiliary Head (set regression loss weight w_reg = 0).
     8. No oversampling (set oversample_method = "none").
     9. Standard SMOTE instead of SMOTENC (to show the impact of not handling categoricals correctly).
     10. No Class-Balanced Focal Loss (use standard Cross Entropy loss instead).
   - For each variant, run 5-fold cross-validation on the training pool and calculate the average F1-Macro, Recall-Low, and Accuracy on validation folds.
   - Save the comparisons to `outputs/v27/ablation_results.csv` with columns: `variant`, `f1_macro`, `recall_low`, `accuracy`.

3. Run the scripts:
   - Run `scripts/run_v27_ensemble.py` for all three datasets to generate the ensemble metrics files.
   - Run `scripts/run_v27_ablation.py` to generate the ablation study results.

MANDATORY INTEGRITY WARNING:
> DO NOT CHEAT. All implementations must be genuine. DO NOT
> hardcode test results, create dummy/facade implementations, or
> circumvent the intended task. A Forensic Auditor will independently
> verify your work. Integrity violations WILL be detected and your
> work WILL be rejected.

Write a complete report of the ensembling results, ablation results, and verify that the output files exist. Write the report to `c:\Huflit\kltn\.agents\teamwork_preview_worker_ensembling_v27_1\handoff.md` and send a message when complete.
