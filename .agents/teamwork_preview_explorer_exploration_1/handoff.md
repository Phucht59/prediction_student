# Handoff Report

## 1. Observation
- **RuleBasedLearningPathEngine**: Located in `src/explainability.py` (lines 107-224). The main recommendation generator is the `generate` method (lines 195-223) which relies on `_student_risks` (lines 123-146) and `_student_actions` (lines 169-180) for student datasets, and `_xapi_risks` (lines 148-167) and `_xapi_actions` (lines 182-193) for xapi datasets.
- **Datasets**: Three datasets exist in `data/raw/`:
  - `student-mat.csv` (56,993 bytes, semicolon delimited)
  - `student-por.csv` (93,220 bytes, semicolon delimited)
  - `xAPI-Edu-Data.csv` (38,026 bytes, comma delimited)
- **Class Binning/Target Mapping**:
  - Student datasets target `G3` mapped to classes Low (0-9), Medium (10-14), High (15-20) in `src/data_pipeline.py` (lines 24-25).
  - xAPI dataset target `Class` mapped to Low (0), Medium (1), High (2) in `src/data_pipeline.py` (line 32) using `XAPI_CLASS_MAPPING` from `src/config.py` (line 61).
- **Pipeline Orchestration**: Coordinated by `scripts/run_pipeline.py`. Includes Optuna hyperparameter optimization (`load_study` lines 92-133), ensemble seed training over 5 fixed seeds (`train_seed_ensemble` lines 191-321), metrics calculation (`calculate_metrics` lines 324-332), feature importance permutation (`explain_model` lines 443-450), and PostgreSQL persistence (`persist_evaluation_to_postgres` lines 40-200).
- **Test Suite Execution**: Run with:
  ```powershell
  C:\Users\THPhu\anaconda3\envs\kltn\python.exe -m pytest -v
  ```
  Resulted in **9 passed, 1 failed**:
  - Failed test: `tests/test_thesis_pipeline.py::test_forbidden_architectures_and_losses_are_removed`
  - Error snippet:
    ```text
    E           assert 'FocalLoss' not in '"""CNN-BiLS...im,\n    )\n'
    E             
    E             'FocalLoss' is contained here:
    ...
    tests\test_thesis_pipeline.py:106: AssertionError
    ```
  - `FocalLoss` is defined in `src/models.py` (line 12): `class FocalLoss(nn.Module):` and used in `src/train_pipeline.py` (lines 25, 265, 318).

## 2. Logic Chain
- The test `test_forbidden_architectures_and_losses_are_removed` is designed to verify that "FocalLoss" is not present in the codebase (specifically `src/models.py`), enforcing the thesis constraint that only CNN, Bi-LSTM, and MLP are used without Focal Loss (Observation: `tests/test_thesis_pipeline.py` line 106).
- The class `FocalLoss` is defined in `src/models.py` (line 12) and imported/referenced in `src/train_pipeline.py` (Observation: `src/models.py` line 12, `src/train_pipeline.py` lines 25, 265, 318).
- Consequently, running the test suite causes `test_forbidden_architectures_and_losses_are_removed` to assert false and fail (Observation: test execution result).
- Due to the read-only constraint of this task (Observation: Key constraint "Read-only investigation — do NOT implement"), the explorer must not modify the code to fix the failure.
- Therefore, the failure is reported as-is, highlighting an architectural violation in the current baseline implementation that should be resolved by the implementer.

## 3. Caveats
- End-to-end database integration was not tested with a live PostgreSQL instance. It was verified at the code level (`src/evaluation.py` and `database/schema.sql`).
- The actual model training and Optuna parameter tuning were not executed for full epochs/trials to prevent modifying saved model artifacts on disk.

## 4. Conclusion
- The recommendation engine is currently rule-based, mapping student features to risk priorities, risk bands, risk factor logs, and staged learning path actions.
- The pipeline utilizes a structured 80/20 train-test split, Optuna optimization, and a 5-seed ensemble with SWA.
- An architectural constraint is currently violated: `FocalLoss` is defined and used in `src/models.py` and `src/train_pipeline.py`, causing `test_forbidden_architectures_and_losses_are_removed` to fail. This must be corrected before migrating to a PyTorch MLP model.

## 5. Verification Method
- **Environment**: Use the conda environment located at `C:\Users\THPhu\anaconda3\envs\kltn`.
- **Command**: Run the test suite using:
  ```powershell
  C:\Users\THPhu\anaconda3\envs\kltn\python.exe -m pytest -v
  ```
- **Expected Result**: 9 tests pass, and `test_forbidden_architectures_and_losses_are_removed` fails due to the presence of `FocalLoss` in `src/models.py`.
