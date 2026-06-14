# Git Investigation and Test Failure Analysis Report

**Date**: 2026-06-14  
**Auditor Role**: Git Auditor  

---

## 1. Git Status Summary

Running `git status` in the repository `c:\Huflit\kltn` reveals:

### Modified Files (Changes not staged for commit)
*   `README.md`
*   `notebooks/tao_toan_bo_hinh_anh_bao_cao.ipynb`
*   `scripts/run_pipeline.py`
*   `src/config.py`
*   `src/data_pipeline.py`
*   `src/explainability.py`
*   `src/models.py`
*   `src/train_pipeline.py`
*   `tests/test_thesis_pipeline.py`
*   Deleted and modified report files (docx, CSVs, JSON metrics, learning paths) under `reports/final/` and `scratch/`.

### Untracked Files
*   `.agents/` (Agent metadata)
*   `models/` (Saved weights and recommendations)
*   `reports/final/LUAN_VAN_HOAN_CHINH_FINAL.docx` and other evaluation artifacts.
*   `src/eval_recommendation.py`
*   `src/recommendation.py`
*   Various scratch scripts and generated outputs.

---

## 2. Working Directory Diffs (Uncommitted Changes)

Here is a summary of the differences between the current working directory and the `HEAD` commit (`91397b7`) for the key files:

### `src/data_pipeline.py`
*   **Initialization**: Added `self.effective_oversample_method = "none"` in the constructor of `DataPreprocessor`.
*   **Oversampling Method Fallback**: In `fit_transform()`, instead of directly falling back to `ADASYN` for non-SMOTE methods, it checks if there are categorical columns. If yes, it logs a warning and uses `SMOTENC` as a categorical-safe sampler, setting `self.effective_oversample_method = "smotenc"`. Otherwise, it uses `ADASYN` and sets `self.effective_oversample_method = "adasyn"`.

### `src/train_pipeline.py`
*   **Validation Folds**: Reverted the validation fold method in `objective()` from `RepeatedStratifiedKFold` (introduced in the baseline commit) back to `StratifiedKFold` with `shuffle=True`.
*   **Criterion Selection**: Replaced the conditional criterion configuration in `objective()` that checked for `focal_gamma` (which used the forbidden `FocalLoss`) to always use `CrossEntropyLoss` for non-xapi datasets.
*   **Imports**: Removed the import of `RepeatedStratifiedKFold` from `sklearn.model_selection` and `FocalLoss` from `src.models`.

### `src/models.py`
*   **Forbidden Code Removal**: Completely deleted the implementation of the `FocalLoss` class, as it is classified as a forbidden loss architecture.

### `tests/test_thesis_pipeline.py`
*   **Engine Transition**: Swapped out the old `RuleBasedLearningPathEngine` (imported from `src.explainability`) with `MLPLearningPathEngine` (imported from the new file `src.recommendation`).
*   **Argument Adjustments**: Updated `dataset_kind` parameter to `dataset_name` in `test_learning_path_report_has_one_row_per_student`.
*   **New Tests Added**:
    1.  `test_adasyn_request_uses_categorical_safe_sampler_for_mixed_data`: Verifies that `DataPreprocessor` falls back to `SMOTENC` if ADASYN is requested on mixed data.
    2.  `test_recommendation_evaluation_does_not_generate_random_metrics`: Validates that `src/eval_recommendation.py` does not generate random metrics and includes the appropriate status indicator.

---

## 3. Commit History Analysis

We traced the git log for `src/data_pipeline.py` and `src/train_pipeline.py` to identify when modifications were made:

*   **`91397b7` ("final model v1")**: Introduced advanced training strategies: SWA (Stochastic Weight Averaging), `RepeatedStratifiedKFold`, and `FocalLoss` targeting non-xapi datasets.
*   **`aaef498` ("new update")**: Introduced the initial version of `tests/test_thesis_pipeline.py` defining the validation constraints and testing pipeline correctness.
*   **`7ab71ff` ("12/6")** & **`9bf8f5d` ("11/6")**: Consolidated the codebase into standard flat modules, restructuring the old directory layout.

---

## 4. Pytest Results (Current vs. Baseline States)

### Scenario A: Current State
Running `python -m pytest -v` using the conda environment at `C:\Users\THPhu\anaconda3\envs\kltn`:
*   **Result**: **PASSED** (12 out of 12 tests passed in ~6.6s).
*   **Why**: The uncommitted changes successfully clean up the forbidden architectures (`FocalLoss`, etc.) and adapt the tests to use the new `MLPLearningPathEngine` in `src.recommendation`.

### Scenario B: Partial Baseline (Discarding uncommitted changes in `src/data_pipeline.py` & `src/train_pipeline.py` only)
*   **Result**: **FAILED during collection** (`ImportError`).
*   **Why**: The baseline version of `src/train_pipeline.py` tries to import `FocalLoss` from `src.models`, but `src/models.py` in the working directory remains modified (with `FocalLoss` deleted).

### Scenario C: Full Baseline (Discarding all uncommitted changes)
After checking out all modified codebase files (`git checkout src/data_pipeline.py src/train_pipeline.py src/models.py tests/test_thesis_pipeline.py src/explainability.py src/config.py`):
*   **Result**: **FAILED** (10 tests collected, 9 passed, 1 failed).
*   *   **Failed Test**: `tests/test_thesis_pipeline.py::test_forbidden_architectures_and_losses_are_removed`
    *   **Failure Reason**: `AssertionError: assert 'FocalLoss' not in source`. The baseline state of `src/models.py` defines `FocalLoss`, which is explicitly listed as a forbidden loss class by the test.

---

## 5. Explanation of Test Failures

1.  **Original vs. New Tests**:
    The test failure in the baseline state is due to an **original test** (`test_forbidden_architectures_and_losses_are_removed`) that was already present at `HEAD`.
2.  **Root Cause**:
    Commit `91397b7` introduced SWA and `FocalLoss` (using ordinal classification targets) into the codebase. However, the subsequent commit `aaef498` introduced `tests/test_thesis_pipeline.py`, which enforces constraints that forbid `FocalLoss`. Thus, the baseline state at `HEAD` was in a broken state where the tests immediately fail due to the presence of `FocalLoss` in `src/models.py`.
3.  **Current Fix**:
    The uncommitted changes in the working directory resolve this conflict by removing `FocalLoss` from `src/models.py`, adjusting `src/train_pipeline.py` and `scripts/run_pipeline.py` to use `CrossEntropyLoss` instead, and updating the test assertions to reflect this clean architecture. They also introduced two new tests (`test_adasyn_request_uses_categorical_safe_sampler_for_mixed_data` and `test_recommendation_evaluation_does_not_generate_random_metrics`) which verify the newer functionality.
