# Git Investigation & Verification Report

This report summarizes the findings of the git repository status, diffs, history, and test suite verification for the target files: `src/data_pipeline.py` and `src/train_pipeline.py`.

---

## 1. Observation

### Current Git Status
Running `git status` in the repository root `c:\Huflit\kltn` shows that there are multiple uncommitted changes:
```
Changes not staged for commit:
  (use "git add/rm <file>..." to update what will be committed)
  (use "git restore <file>..." to discard changes in working directory)
	...
	modified:   src/data_pipeline.py
	modified:   src/train_pipeline.py
	modified:   src/models.py
	modified:   tests/test_thesis_pipeline.py
	...
```

### Exact Uncommitted Changes

#### `src/data_pipeline.py`
Running `git diff src/data_pipeline.py` reveals the following modifications:
```diff
diff --git a/src/data_pipeline.py b/src/data_pipeline.py
index 42b8f9e..55eb3d0 100644
--- a/src/data_pipeline.py
+++ b/src/data_pipeline.py
@@ -247,6 +247,7 @@ class DataPreprocessor:
         self.scalers = {}
         self.label_encoders = {}
         self.target_encoder = LabelEncoder()
+        self.effective_oversample_method = "none"
         
     def fit_transform(self, df: pd.DataFrame):
         """Fit on train pool and transform it. Also handles SMOTE/ADASYN."""
@@ -303,18 +304,35 @@ class DataPreprocessor:
                         random_state=42,
                         k_neighbors=effective_k_neighbors,
                     )
+                    self.effective_oversample_method = "smotenc"
                 else:
                     sampler = SMOTE(
                         sampling_strategy=strategy,
                         random_state=42,
                         k_neighbors=effective_k_neighbors,
                     )
+                    self.effective_oversample_method = "smote"
             else:
-                sampler = ADASYN(
-                    sampling_strategy=strategy,
-                    random_state=42,
-                    n_neighbors=effective_k_neighbors,
-                )
+                cat_indices = [X.columns.get_loc(c) for c in self.categorical_cols] if self.categorical_cols else []
+                if cat_indices:
+                    logger.warning(
+                        "ADASYN cannot preserve label-encoded categorical values; "
+                        "using categorical-safe SMOTENC for this mixed dataset."
+                    )
+                    sampler = SMOTENC(
+                        categorical_features=cat_indices,
+                        sampling_strategy=strategy,
+                        random_state=42,
+                        k_neighbors=effective_k_neighbors,
+                    )
+                    self.effective_oversample_method = "smotenc"
+                else:
+                    sampler = ADASYN(
+                        sampling_strategy=strategy,
+                        random_state=42,
+                        n_neighbors=effective_k_neighbors,
+                    )
+                    self.effective_oversample_method = "adasyn"
             try:
                 X_resampled, y_resampled = sampler.fit_resample(X, y_encoded)
                 X = pd.DataFrame(X_resampled, columns=X.columns)
```

#### `src/train_pipeline.py`
Running `git diff src/train_pipeline.py` reveals the following modifications:
```diff
diff --git a/src/train_pipeline.py b/src/train_pipeline.py
index 053472a..8a12d5d 100644
--- a/src/train_pipeline.py
+++ b/src/train_pipeline.py
@@ -11,7 +11,7 @@ import torch
 import torch.nn as nn
 import torch.optim as optim
 from sklearn.metrics import accuracy_score, f1_score
-from sklearn.model_selection import StratifiedKFold, RepeatedStratifiedKFold
+from sklearn.model_selection import StratifiedKFold
 from torch.utils.data import DataLoader
 
 from src.config import TrainingConfig
@@ -22,7 +22,7 @@ from src.data_pipeline import (
     apply_feature_engineering,
     get_sequence_columns,
 )
-from src.models import create_model, FocalLoss
+from src.models import create_model
 from src.utils import set_seed, setup_logger
 
 logger = setup_logger("train_pipeline")
@@ -253,8 +253,7 @@ def objective(trial, df_train_pool: pd.DataFrame, spec, target_mode: str, cv_fol
     smote_ratio = params["smote_ratio"]
     model_config = params
 
-    # Strategy 3: Repeated Stratified K-Fold for more robust validation
-    stratified_folds = RepeatedStratifiedKFold(n_splits=cv_folds, n_repeats=3, random_state=42)
+    stratified_folds = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
     target = df_train_pool[spec.target_col].astype(int).to_numpy()
     fold_f1s = []
     sequence_columns = get_sequence_columns(spec.kind)
@@ -314,10 +313,7 @@ def objective(trial, df_train_pool: pd.DataFrame, spec, target_mode: str, cv_fol
         if spec.kind == "xapi":
             criterion = nn.BCEWithLogitsLoss()
         else:
-            if "focal_gamma" in model_config:
-                criterion = FocalLoss(weight=class_weights, gamma=model_config["focal_gamma"])
-            else:
-                criterion = nn.CrossEntropyLoss(weight=class_weights)
+            criterion = nn.CrossEntropyLoss(weight=class_weights)
         optimizer = optim.Adam(
             model.parameters(),
             lr=learning_rate,
```

### Git History
Running `git log -n 5 --pretty=format:"%h - %an, %ad : %s"` for both files shows they share the exact same commit history of recent modifications:
- **`91397b7`** - Phucht59, Sat Jun 13 20:51:48 2026 +0700 : final model v1
- **`aaef498`** - Phucht59, Sat Jun 13 03:28:52 2026 +0700 : new update
- **`7ab71ff`** - Phucht59, Fri Jun 12 14:22:06 2026 +0700 : 12/6
- **`9bf8f5d`** - Phucht59, Thu Jun 11 08:30:50 2026 +0700 : 11/6

Commit `91397b7` introduced:
1. Support for `SMOTENC` inside `SMOTE` block if categorical columns exist in `src/data_pipeline.py`.
2. Usage of `RepeatedStratifiedKFold` and custom `FocalLoss` in `src/train_pipeline.py`.

The uncommitted changes currently in the working directory represent a refactoring to:
1. Extend `SMOTENC` fallback to `ADASYN` requests when mixed features exist.
2. Track the effective oversample method using `self.effective_oversample_method`.
3. Revert `RepeatedStratifiedKFold` back to `StratifiedKFold` and remove `FocalLoss` imports/usage from `src/train_pipeline.py` (matching the deletion of `FocalLoss` in `src/models.py`).

### Test Suite Execution
- **Under Current State**: Running `py -3.10 -m pytest` succeeds.
  ```
  tests\test_thesis_pipeline.py ............                               [100%]
  ============================= 12 passed in 15.52s =============================
  ```
- **With `src/data_pipeline.py` Changes Discarded**: Running `git checkout src/data_pipeline.py` and running the test suite causes a test failure.
  ```
  tests\test_thesis_pipeline.py ....F.......                               [100%]
  ================================== FAILURES ===================================
  ______ test_adasyn_request_uses_categorical_safe_sampler_for_mixed_data _______
  ...
          preprocessor = DataPreprocessor("target", oversample_method="adasyn", resampling_k_neighbors=2)
          transformed = preprocessor.fit_transform(frame)
  >       assert preprocessor.effective_oversample_method == "smotenc"
  E       AttributeError: 'DataPreprocessor' object has no attribute 'effective_oversample_method'
  
  tests\test_thesis_pipeline.py:106: AttributeError
  =========================== short test summary info ===========================
  FAILED tests/test_thesis_pipeline.py::test_adasyn_request_uses_categorical_safe_sampler_for_mixed_data
  ======================== 1 failed, 11 passed in 6.90s =========================
  ```

---

## 2. Logic Chain

1. **Uncommitted updates in tests**: The file `tests/test_thesis_pipeline.py` contains uncommitted changes including the test `test_adasyn_request_uses_categorical_safe_sampler_for_mixed_data`.
2. **Dependence on preprocessing changes**: This test explicitly asserts that if `oversample_method="adasyn"` is requested on a mixed dataset, `DataPreprocessor` falls back to `smotenc` and updates the attribute `effective_oversample_method` (lines 104-107 in `tests/test_thesis_pipeline.py`).
3. **Preprocess modification**: The uncommitted changes in `src/data_pipeline.py` implement this exact safeguard fallback to `SMOTENC` for `ADASYN` and define `self.effective_oversample_method`.
4. **Behavior on Discard**: When `git checkout src/data_pipeline.py` is executed, the preprocessor reverts to its committed state (from commit `91397b7`). In this state, it does not define `effective_oversample_method` nor does it check for categorical features when handling `ADASYN`.
5. **Resulting Failure**: Consequently, running `pytest` raises an `AttributeError` on the undefined attribute, causing the test suite to fail (1 failed, 11 passed).

---

## 3. Caveats

- Tests were run using `py -3.10 -m pytest` because the global `pytest` command is not available in the command environment, whereas Python 3.10 is installed and contains the `pytest` dependency.
- The full execution of the end-to-end pipelines (via `run_all.bat`) was not performed as it executes 350+ hyperparameter tuning trials, which is computationally heavy and outside the read-only scope of this investigation.
- No files were modified permanently; the modified state of `src/data_pipeline.py` was fully restored after verifying the test failure.

---

## 4. Conclusion

- **Git Status**: Currently, there are uncommitted modifications to `src/data_pipeline.py`, `src/train_pipeline.py`, `src/models.py`, and `tests/test_thesis_pipeline.py`.
- **Uncommitted vs Committed**:
  - Committed code (as of `91397b7` on Sat Jun 13 20:51:48 2026) introduced `SMOTENC` fallback for `SMOTE`, but used standard `ADASYN` (non-categorical-safe) and implemented `RepeatedStratifiedKFold` and `FocalLoss`.
  - Uncommitted code implements a `SMOTENC` fallback safeguard for `ADASYN` (with tracking via `effective_oversample_method`) and removes forbidden components (`FocalLoss` and `RepeatedStratifiedKFold`).
- **Test Suit Impact**: The test suite **fails** (1 failure) if the uncommitted changes in `src/data_pipeline.py` are discarded, due to the missing implementation of the `ADASYN` categorical safeguard and its corresponding attribute tracking required by `test_adasyn_request_uses_categorical_safe_sampler_for_mixed_data`.

---

## 5. Verification Method

To independently verify these findings, run the following commands in the repository root directory `c:\Huflit\kltn`:

1. **Check Status**:
   ```powershell
   git status
   ```
   *Verify that `src/data_pipeline.py`, `src/train_pipeline.py`, `src/models.py`, and `tests/test_thesis_pipeline.py` are listed as modified.*

2. **Verify Current Test Pass**:
   ```powershell
   py -3.10 -m pytest
   ```
   *Expect: 12 passed.*

3. **Verify Failure after Discarding Changes**:
   Run the following script to copy, discard changes, test, and restore:
   ```powershell
   Copy-Item src/data_pipeline.py src/data_pipeline.py.backup
   git checkout src/data_pipeline.py
   py -3.10 -m pytest
   Move-Item -Force src/data_pipeline.py.backup src/data_pipeline.py
   ```
   *Expect: 1 failed (AttributeError on `effective_oversample_method`), 11 passed.*
