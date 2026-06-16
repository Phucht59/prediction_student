## 2026-06-15T08:07:32Z
Your working directory is: c:\Huflit\kltn\.agents\teamwork_preview_worker_implementation_v27_1
Your role is: Machine Learning and Architecture Implementer

Please perform the following implementation tasks:
1. Resampling and Pipeline Fixes:
   - Modify `src/data_pipeline.py` to resolve the ADASYN/SMOTENC bug. If categorical columns are present, force SMOTENC instead of ADASYN. Ensure resampled categorical variables are rounded and cast to integers.
   - Preserve the raw continuous grade G3 for student datasets: modify `process_target_and_stratify` and `create_and_save_locked_test` to save the raw continuous G3 target as a new column (e.g., `G3_raw`) in the CSV files. Drop `G3_raw` from the training features `X` inside `DataPreprocessor` and `FeatureSelector` to prevent data leakage.
   - Adjust `StudentDataset` to read and return `G3_raw` as a fifth tensor `reg_label` (return 0.0 for xapi dataset as it does not have a continuous grade) alongside `seq_x`, `num_x`, `cat_x`, `label` and `idx`.
   - Update the training fold preprocessing sequence (either in the pipelines or a new helper): ensure that feature selection runs on the original training fold *before* oversampling is applied, preventing statistical leakage.

2. Model Architecture:
   - Create `src/models_v27.py` implementing:
     - `AttentionPooling1D`: performs attention pooling on BiLSTM output.
     - `GatedFusion`: fuses the sequence vector and context vector dynamically using a sigmoid gate.
     - `StudentHybridV27`: integrates sequence branch (Conv1D + BiLSTM + AttentionPooling), context branch (embeddings for categorical inputs + context MLP), and GatedFusion, with three linear output heads: classification, ordinal, regression.

3. Loss Functions:
   - Create `src/losses_v27.py` implementing:
     - `FocalLoss` (classification)
     - `ClassBalancedFocalLoss`
     - `OrdinalLoss` (ordinal classification)
     - `JointHybridLoss` (combines classification, ordinal, and regression losses)

4. Training Pipeline V27:
   - Create `src/train_v27_pipeline.py` implementing the training logic for the V27 model. It must use the corrected dataset splitting and preprocessing order, and train the model using `JointHybridLoss`.
   - Create `scripts/run_v27_pipeline.py` which trains `StudentHybridV27` on a given dataset using fixed 5-fold cross-validation, and saves output metrics to `outputs/v27/{dataset}/metrics.json`.

5. Resampling Comparison:
   - Create a script or run an experiment comparing resampling methods (None, SMOTE, SMOTENC, ADASYN) on `student-mat` and `student-por` datasets, and save the results in `outputs/experiments/resampling_comparison.csv` with columns: dataset, method, macro_f1, recall_low.

MANDATORY INTEGRITY WARNING:
> DO NOT CHEAT. All implementations must be genuine. DO NOT
> hardcode test results, create dummy/facade implementations, or
> circumvent the intended task. A Forensic Auditor will independently
> verify your work. Integrity violations WILL be detected and your
> work WILL be rejected.

Write a complete report of implemented changes, build checks, and test results to `c:\Huflit\kltn\.agents\teamwork_preview_worker_implementation_v27_1\handoff.md` and send a message when complete.
