# Handoff Report — FocalLoss Bypass Remediation

## 1. Observation
- **Observation 1.1**: The codebase had a dynamic class registration bypass for `FocalLoss` in `src/models/models.py` (lines 11-28):
  ```python
  class Focal_Loss(nn.Module):
      ...
  globals()["Focal" + "Loss"] = Focal_Loss
  ```
  This was exposed in `src/models/__init__.py` using:
  ```python
  from .models import StudentHybridModel, create_model, FocalLoss
  __all__ = ["StudentHybridModel", "create_model", "FocalLoss"]
  ```
- **Observation 1.2**: In `src/train_pipeline.py`, lines 25 and 317-318 imported and instantiated `FocalLoss`:
  ```python
  from src.models import create_model, FocalLoss
  ...
  if "focal_gamma" in model_config:
      criterion = FocalLoss(weight=class_weights, gamma=model_config["focal_gamma"])
  ```
- **Observation 1.3**: In `scripts/run_pipeline.py`, lines 261 and 266-267 imported and instantiated `FocalLoss`:
  ```python
  from src.models import FocalLoss, create_model
  ...
  if "focal_gamma" in best_params:
      criterion = FocalLoss(weight=class_weights, gamma=best_params["focal_gamma"])
  ```
- **Observation 1.4**: pytest unit tests had a forbidden architectures check `test_forbidden_architectures_and_losses_are_removed` which checked for `FocalLoss` in `src/models/models.py`. However, it was bypassed because the code text split the name into `"Focal"` and `"Loss"`.
- **Observation 1.5**: Running `C:\Users\THPhu\anaconda3\envs\kltn\python.exe -m pytest` after applying changes succeeds with all 16 tests passing in 9.20s:
  ```
  tests\test_recommender.py ......                                         [ 37%]
  tests\test_thesis_pipeline.py ..........                                 [100%]
  ============================= 16 passed in 9.20s =============================
  ```
- **Observation 1.6**: Running `C:\Users\THPhu\anaconda3\envs\kltn\python.exe scripts/run_recommender_pipeline.py --dataset student-mat` completed successfully:
  ```
  2026-06-15 00:28:33,563 - run_recommender_pipeline - INFO - RA-HLPR pipeline completed successfully. All outputs saved to outputs/recommender/
  ```

## 2. Logic Chain
- **Step 1**: Deleting `Focal_Loss` definition and `globals()["Focal" + "Loss"] = Focal_Loss` dynamic registration from `src/models/models.py` completely eliminates the class from the model codebase (supported by Observation 1.1).
- **Step 2**: Modifying `src/models/__init__.py` to remove `FocalLoss` from imports and exports prevents importing a non-existent class (supported by Observation 1.1).
- **Step 3**: Modifying `src/train_pipeline.py` to remove the `FocalLoss` import, removing `focal_gamma` from trial suggestions, and replacing the conditional checks that instantiate `FocalLoss` with standard `CrossEntropyLoss` removes all pipeline-level dependencies (supported by Observation 1.2).
- **Step 4**: Modifying `scripts/run_pipeline.py` to remove the import and instantiation of `FocalLoss` ensures standard final training also uses standard `CrossEntropyLoss` (supported by Observation 1.3).
- **Step 5**: Running `pytest` confirms that removing `FocalLoss` completely satisfies the architecture checker `test_forbidden_architectures_and_losses_are_removed` without breaking any other code (supported by Observation 1.4, 1.5).
- **Step 6**: Executing `scripts/run_recommender_pipeline.py` end-to-end verifies that the recommender system remains fully operational without FocalLoss (supported by Observation 1.6).

## 3. Caveats
- No caveats.

## 4. Conclusion
- The architectural concern raised regarding `FocalLoss` dynamic bypass has been remediated. `FocalLoss` is completely expunged from the codebase, standard `CrossEntropyLoss` is used, and the pipeline executes successfully.

## 5. Verification Method
- **Test Command**: Run `C:\Users\THPhu\anaconda3\envs\kltn\python.exe -m pytest` from `c:\Huflit\kltn`. It must pass all 16 tests.
- **Pipeline Execution**: Run `C:\Users\THPhu\anaconda3\envs\kltn\python.exe scripts/run_recommender_pipeline.py --dataset student-mat`. It must finish successfully.
- **Files to Inspect**:
  - `src/models/models.py` to verify that `Focal_Loss` class and dynamic registration line do not exist.
  - `src/models/__init__.py` to verify that `FocalLoss` is not exported.
  - `src/train_pipeline.py` and `scripts/run_pipeline.py` to verify that `FocalLoss` is not imported or instantiated.
- **Invalidation Conditions**: If `FocalLoss` is found in `src/models/models.py`, or if `test_forbidden_architectures_and_losses_are_removed` fails, or if the recommender pipeline fails to run.
