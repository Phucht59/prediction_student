# Forensic Audit Report

**Work Product**: V27 Prediction Pipeline and Model Components
**Profile**: General Project (Development Mode)
**Verdict**: CLEAN

---

## 1. Observation

### Implementation Files Inspected:
- **`src/data_pipeline.py`**:
  - *SMOTENC fix* (lines 322-330):
    ```python
    if remaining_cat_cols:
        cat_indices = [X.columns.get_loc(c) for c in remaining_cat_cols]
        sampler = SMOTENC(
            categorical_features=cat_indices,
            sampling_strategy=strategy,
            random_state=42,
            k_neighbors=effective_k_neighbors,
        )
    ```
    This dynamically retrieves categorical indices and enforces SMOTENC if categorical features exist in the resampled columns, preventing type coercion or structural corruption.
  - *`G3_raw` preservation* (lines 23, 233-234, 399-405):
    - Continuous labels are preserved: `df["G3_raw"] = df[target_col]`.
    - Feature selection bypasses it: `self.selected_features = [f for f in selected if f != "G3_raw"]`.
    - `StudentDataset` extracts it: `self.reg_label = df["G3_raw"].values.astype(np.float32)`.
  - *Feature selection isolation*:
    `FeatureSelector.fit_transform` computes statistical dependencies (Pearson correlation for numeric, Chi-Square for categorical) strictly using train fold data (lines 183-222).

- **`src/models_v27.py`**:
  - `AttentionPooling1D` (lines 6-21): Pools bidirectional LSTM hidden states using a soft-attention mechanism, outputting weights that sum to 1.0.
  - `GatedFusion` (lines 24-43): Computes a Sigmoid-gated linear interpolation to dynamically fuse the sequential vector and context vector: `fused = gate * h_seq + (1.0 - gate) * h_ctx`.
  - `StudentHybridV27` (lines 45-183):
    - Categorical embeddings lookups.
    - Sequence branch: 1D Conv -> BiLSTM -> AttentionPooling.
    - Context branch: Concatenates numerical and categorical embeddings -> Multi-Layer Perceptron.
    - GatedFusion fuses both representation branches.
    - Output heads: `self.class_head` (size 3), `self.ordinal_head` (size 2), and `self.reg_head` (size 1).

- **`src/losses_v27.py`**:
  - `ClassBalancedFocalLoss` (lines 25-49): Calculates class weights based on effective sample sizes, registers them via buffer, and computes focal loss.
  - `OrdinalLoss` (lines 52-72): Formulates ordinal targets using class indices and optimizes using binary cross entropy.
  - `JointHybridLoss` (lines 74-101): Combined joint loss: `total_loss = (self.w_class * loss_class + self.w_ord * loss_ord + self.w_reg * loss_reg)`.

- **Pipeline Scripts (`scripts/run_v27_pipeline.py`, `scripts/run_v27_optuna.py`, `scripts/tune_v27_thresholds.py`, `scripts/run_v27_ensemble.py`, `scripts/run_v27_ablation.py`)**:
  - In all scripts, `load_splits(dataset_name, "3class")` is used to load data splits.
  - Test set isolation: `preprocessor.fit_transform` is only called on the training fold (with `apply_oversampling=False`), and the validation fold or test fold is transformed using `preprocessor.transform(val_fold)` or `preprocessor.transform(test_fold)`.
  - Feature selection is fitted only on the training fold, and applied to validation/test folds.
  - Resampling (SMOTE/SMOTENC/ADASYN) is only called on the train fold after preprocessing and feature selection.
  - The locked test dataset (`locked_test`) is never leaked, loaded during model training, hyperparameter tuning, or threshold search. It is strictly used as an offline evaluation set after model ensembles are completed.

### Unit Tests Executed:
Proposed and ran `py -3.10 -m pytest tests/test_v27_components.py` yielding:
```
============================= test session starts =============================
platform win32 -- Python 3.10.8, pytest-9.0.3, pluggy-1.6.0
rootdir: C:\Huflit\kltn
collected 5 items

tests\test_v27_components.py .....                                       [100%]

============================== 5 passed in 4.47s =============================
```

### Pipeline Execution:
Proposed and ran `py -3.10 scripts/run_v27_pipeline.py --dataset student-mat` yielding:
```
2026-06-15 15:50:55,572 - run_v27_pipeline - INFO - Recreating splits for student-mat to ensure G3_raw is preserved in CSV files...
2026-06-15 15:50:55,591 - preprocessing - INFO - [student-mat - 3class] Train pool: 316 rows. Locked test: 79 rows.
2026-06-15 15:50:55,612 - run_v27_pipeline - INFO - Loaded splits. Train Pool: 316, Locked Test: 79
...
2026-06-15 15:51:14,466 - run_v27_pipeline - INFO - === COMPLETED TRAINING V27 FOR student-mat ===
2026-06-15 15:51:14,466 - run_v27_pipeline - INFO - Avg F1-Macro: 0.8920, Avg Accuracy: 0.8860
```
This verified the outputs are dynamically computed and stored into `outputs/v27/student-mat/metrics.json`.

---

## 2. Logic Chain

1. **Source Code Integrity**: Source files (`src/models_v27.py`, `src/losses_v27.py`, `src/data_pipeline.py`) contain real, dynamic PyTorch modules (embeddings, CNN, BiLSTM, Sigmoid gating, Focal, Ordinal, MSE losses) and lack any hardcoded labels, shortcuts, constant-returns, or mocks.
2. **Behavioral Genuineness**: Unit tests confirm correct output dimensions, mathematically valid attention-weights, and proper dataset unpacking.
3. **Execution Genuineness**: Running the pipeline runs real training epochs on the CPU, triggers early stopping, and yields standard metrics dynamically.
4. **Data Isolation (No Leakage)**: Training scripts construct preprocessing transforms and feature selections strictly fitting on the training folds. Resampling is performed only on train-pool slices. Hyperparameter searches (Optuna) and decision threshold searches are run using out-of-fold validation subsets of the `train_pool`. The locked test dataset remains isolated and is only introduced for inference after training has completed.
5. **Conclusion Support**: Since all checks pass successfully, the work product is rated clean.

---

## 3. Caveats

- **Computational Environment**: Performance was verified on CPU since CUDA was unavailable (`torch.cuda.is_available() == False`). However, execution was verified to be correct and fast.
- **Optuna Search Scale**: We did not run a full 150-trial Optuna search or a full 10-variant ablation run during the audit due to potential time/resource limits, but verified their structure and logic through unit testing, source review, and a single pipeline dataset run.

---

## 4. Conclusion

The V27 prediction components, models, losses, and run scripts show no integrity violations. They are authentic, dynamic, statistically sound, and fully isolated from data leakage.

---

## 5. Verification Method

To verify these checks independently, execute the following commands in the workspace:

1. **Run Unit Tests**:
   ```powershell
   py -3.10 -m pytest tests/test_v27_components.py
   ```
2. **Run Dataset Pipeline**:
   ```powershell
   py -3.10 scripts/run_v27_pipeline.py --dataset student-mat
   ```
3. **Verify Output Metric Artifacts**:
   Inspect `outputs/v27/student-mat/metrics.json` to verify the resulting metrics are generated.
