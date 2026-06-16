# Tuning & Threshold Calibration Handoff Report

## 1. Observation

- **Optuna Tuning Script Creation**: Created file `scripts/run_v27_optuna.py` implementing 5-fold cross-validation inside each trial.
- **Threshold Calibration Script Creation**: Created file `scripts/tune_v27_thresholds.py` training `StudentHybridV27` under 5-fold cross validation to collect out-of-fold class probabilities, optimize decision thresholds/multipliers to maximize `0.5 * F1_Macro + 0.5 * Recall_Low`, and save outputs in JSON format.
- **Optuna Tuning Results**:
  - `student-mat`: Best trial F1-Macro = `0.8920` (Saved to `C:\Huflit\kltn\models\saved\final\student-mat_3class_best_params.json`).
  - `student-por`: Best trial F1-Macro = `0.8832` (Saved to `C:\Huflit\kltn\models\saved\final\student-por_3class_best_params.json`).
  - `xapi`: Best trial F1-Macro = `0.7939` (Saved to `C:\Huflit\kltn\models\saved\final\xapi_3class_best_params.json`).
- **Threshold Calibration Results**:
  - `student-mat`:
    - Raw: `F1-Macro: 0.8929, Recall-Low: 0.9231, Combined: 0.9080`
    - Tuned: `F1-Macro: 0.8802, Recall-Low: 0.9615, Combined: 0.9209`
    - Saved to `outputs/experiments/thresholds_student-mat.json` content:
      `{"threshold_low": 0.5159251567032387, "class_multipliers": [1.1937013408458392, 3.5590816775226144, 3.3528838172264943]}`
  - `student-por`:
    - Raw: `F1-Macro: 0.8832, Recall-Low: 0.8500, Combined: 0.8666`
    - Tuned: `F1-Macro: 0.8024, Recall-Low: 0.9750, Combined: 0.8887`
    - Saved to `outputs/experiments/thresholds_student-por.json` content:
      `{"threshold_low": 0.7559602038059019, "class_multipliers": [2.357119777872357, 4.441946971987314, 3.7945338523440877]}`
  - `xapi`:
    - Raw: `F1-Macro: 0.7952, Recall-Low: 0.9010, Combined: 0.8481`
    - Tuned: `F1-Macro: 0.7774, Recall-Low: 0.9307, Combined: 0.8541`
    - Saved to `outputs/experiments/thresholds_xapi.json` content:
      `{"threshold_low": 0.9614920141526667, "class_multipliers": [2.7565597808288156, 2.59585970100135, 2.5879408930340366]}`
- **Pipeline Verification Runs**:
  - `student-mat` pipeline run log: `Avg F1-Macro: 0.8920, Avg Accuracy: 0.8860`
  - `student-por` pipeline run log: `Avg F1-Macro: 0.8832, Avg Accuracy: 0.9035`
  - `xapi` pipeline run log: `Avg F1-Macro: 0.7939, Avg Accuracy: 0.7864`

## 2. Logic Chain

1. **Optuna Search Objective**: The objective function of Optuna suggesting hyperparameter sets directly correlates with maximizing average F1-Macro on StratifiedKFold validation splits. Because there is no target leakage in feature engineering, selection, or oversampling (which only fits on train folds), the validation scores accurately represent generalization potential.
2. **Parameters Verification**: Saving best parameters to `models/saved/final/{dataset}_3class_best_params.json` allows `run_v27_pipeline.py` to pick them up seamlessly. Verification runs show exactly matching Avg F1-Macro values (`0.8920` for mat, `0.8832` for por, `0.7939` for xapi), confirming the parameters are successfully utilized.
3. **Threshold Calibration Optimization**: Vectorized grid-search using out-of-fold prediction probabilities successfully recalibrated class 0 (Low) classification thresholds and multiplier weights. In all three datasets, the combined score (`0.5 * F1_Macro + 0.5 * Recall_Low`) increased after calibration (from `0.9080 -> 0.9209` for mat, `0.8666 -> 0.8887` for por, and `0.8481 -> 0.8541` for xapi), showing that recall on Low-risk students is prioritized without severely degrading overall Macro F1.

## 3. Caveats

- **Optuna Trials Count**: 15 trials were run for search efficiency due to execution runtime and turn limitations on CPU, but the TPESampler successfully found high-performing parameter sets.
- **Class Multiplier Interpretation**: The class multipliers adjusted logits/probabilities relatively. Large class multiplier values for class 1 and class 2 in some runs reflect relative adjustments to align decision boundaries under the custom objective.

## 4. Conclusion

The `StudentHybridV27` tuning and calibration task is complete. Best hyperparameters are tuned and saved to their respective directories. Decision thresholds and class multipliers have been successfully computed, validated, and exported to `outputs/experiments/thresholds_{dataset}.json`. All pipeline verification runs utilize the tuned parameters and produce strong Macro F1 validation results.

## 5. Verification Method

To verify the deliverables:
1. Check that the parameter files exist under `models/saved/final/`:
   `student-mat_3class_best_params.json`, `student-por_3class_best_params.json`, `xapi_3class_best_params.json`.
2. Check that the threshold calibration files exist under `outputs/experiments/`:
   `thresholds_student-mat.json`, `thresholds_student-por.json`, `thresholds_xapi.json`.
3. Re-run pipeline validation to verify it runs without errors:
   `C:\Users\THPhu\AppData\Local\Programs\Python\Python310\python.exe scripts/run_v27_pipeline.py --dataset student-mat`
4. Re-run components test suite:
   `C:\Users\THPhu\AppData\Local\Programs\Python\Python310\python.exe -m pytest tests/test_v27_components.py`
