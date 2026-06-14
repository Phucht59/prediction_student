# Forensic Audit Report

**Work Product**: Preprocessing and resampling logic in `src/data_pipeline.py` and `src/train_pipeline.py`
**Profile**: General Project
**Verdict**: INTEGRITY VIOLATION

---

## 1. Observation

During my investigation of the repository at `c:\Huflit\kltn`, I directly observed the following:

1. **Uncommitted Diffs in Working Directory**:
   Running `git status` in the repository root shows both files are modified:
   ```
   Changes not staged for commit:
       modified:   src/data_pipeline.py
       modified:   src/train_pipeline.py
   ```

2. **Verbatim modifications in `src/data_pipeline.py`**:
   The local uncommitted diff for `src/data_pipeline.py` shows:
   ```python
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
   ```

3. **Verbatim modifications in `src/train_pipeline.py`**:
   The local uncommitted diff for `src/train_pipeline.py` shows:
   ```python
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
   ```

4. **Commit History Diffs**:
   A review of `git log` shows that both files were introduced in commit `9bf8f5d6` ("11/6") and have since evolved significantly.
   Specifically, compared to their initial state at commit `9bf8f5d6`:
   - In `src/data_pipeline.py`:
     - The preprocessing class was renamed from `V26Preprocessor` to `DataPreprocessor`.
     - Parameters `smote_ratio` and `resampling_k_neighbors` were introduced.
     - Resampling logic was updated to use a custom dynamic class balance `strategy` dict instead of default balance.
     - SMOTE was modified to check for categorical columns and dynamically fallback to `SMOTENC` if `cat_indices` is present.
   - In `src/train_pipeline.py`:
     - The Optuna parameter search space for oversampling methods was reduced from `["none", "smote", "adasyn"]` to strictly `["adasyn"]` (for student datasets) and `["smote"]` (for xapi dataset).
     - Validation splits changed from `StratifiedKFold` to `RepeatedStratifiedKFold` (and then back to `StratifiedKFold` in the current working directory).
     - Custom `HybridLoss` and `FocalLoss` were replaced.

5. **Unit Test Results**:
   Running the command `C:\Users\THPhu\anaconda3\envs\kltn\python.exe -m pytest -v` succeeded completely:
   ```
   ============================= 12 passed in 7.02s ==============================
   ```
   However, the test suite `tests/test_thesis_pipeline.py` contains tests that actively assert the *modified* behavior (e.g. `test_adasyn_request_uses_categorical_safe_sampler_for_mixed_data` checks that requesting `adasyn` returns `smotenc` for mixed-type datasets).

---

## 2. Logic Chain

1. The user's request explicitly states:
   - "NO CHANGES were made to the preprocessing or resampling logic in src/data_pipeline.py or src/train_pipeline.py."
   - "Verify that the original resampling algorithm (ADASYN/SMOTENC), casting, and preprocessing steps are 100% identical to their initial state."
2. Observation 1 shows that both `src/data_pipeline.py` and `src/train_pipeline.py` contain uncommitted modifications in the working directory.
3. Observation 2 shows that the uncommitted changes in `src/data_pipeline.py` explicitly alter the resampling algorithm selection by intercepting `adasyn` requests and falling back to `SMOTENC` if categorical columns exist.
4. Observation 3 shows that the uncommitted changes in `src/train_pipeline.py` alter validation fold setup (`RepeatedStratifiedKFold` -> `StratifiedKFold`) and loss criteria (removing `FocalLoss`).
5. Observation 4 shows that historically (comparing `HEAD` or current state with the initial commit `9bf8f5d6` when the pipeline files were created), major changes were introduced, including:
   - Renaming the preprocessor.
   - Introducing config parameters (`smote_ratio`, `resampling_k_neighbors`).
   - Computing dynamic balance strategies for SMOTE/ADASYN.
   - Replacing basic `SMOTE` with `SMOTENC`.
6. Therefore, the resampling algorithm, casting, and preprocessing steps are **not** 100% identical to their initial state.
7. Consequently, the work product contains modifications to these logics, resulting in a verdict of **INTEGRITY VIOLATION**.

---

## 3. Caveats

- I assumed that "initial state" refers to the baseline version of these files when they were first committed in git (`9bf8f5d6`). If the baseline is meant to be the origin branch (`origin/temp-main`), the files did not exist at all in `origin/temp-main` (the codebase was restructured). Under either baseline, the preprocessing and resampling logics have been altered.
- All unit tests passed, but this is because the test suite was updated concurrently to reflect and validate the new resampling fallbacks.

---

## 4. Conclusion

**Verdict**: **INTEGRITY VIOLATION**

The files `src/data_pipeline.py` and `src/train_pipeline.py` have been modified in both the git history and the current working directory. The changes directly impact the resampling logic (introducing dynamic balance strategies, configurations for k-neighbors, and fallbacks to SMOTENC for both SMOTE and ADASYN) and preprocessing classes. Thus, they are not 100% identical to their initial state, which violates the strict constraint.

---

## 5. Verification Method

To independently verify these findings:
1. Check the local status:
   ```bash
   git status
   ```
2. Inspect the diff of `src/data_pipeline.py` to see the SMOTENC fallback implementation:
   ```bash
   git diff src/data_pipeline.py
   ```
3. Inspect the diff of `src/train_pipeline.py` to see changes in validation splits and loss criteria:
   ```bash
   git diff src/train_pipeline.py
   ```
4. Run the project tests to observe that the updated tests pass:
   ```bash
   C:\Users\THPhu\anaconda3\envs\kltn\python.exe -m pytest -v
   ```
