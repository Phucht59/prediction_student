# Handoff Report - explorer_milestone1_1

## 1. Observation
Below are the direct observations from the codebase, configuration, and logs:

* **Predictions Location**:
  * In `src/config.py` line 14:
    ```python
    PREDICTIONS_DIR = REPORTS_DIR / "predictions"
    ```
    where `REPORTS_DIR` is defined as `ROOT_DIR / "reports" / "final"`.
  * In `scripts/run_pipeline.py` lines 360-361:
    ```python
    predictions_path = PREDICTIONS_DIR / f"{args.dataset}_{args.target_mode}_predictions.csv"
    predictions_frame.to_csv(predictions_path, index=False)
    ```
  * Checked directory `c:\Huflit\kltn\reports\final\predictions` and verified the existence of:
    * `student-mat_3class_predictions.csv`
    * `student-mat_3class_v27_predictions.csv`
    * `student-por_3class_predictions.csv`
    * `student-por_3class_v27_predictions.csv`
    * `xapi_3class_predictions.csv`
    * `xapi_3class_v27_predictions.csv`
  * Checked `student-mat_3class_predictions.csv` headers:
    ```csv
    school,sex,age,address,famsize,Pstatus,Medu,Fedu,Mjob,Fjob,reason,guardian,traveltime,studytime,failures,schoolsup,famsup,paid,activities,nursery,higher,internet,romantic,famrel,freetime,goout,Dalc,Walc,health,absences,G1,G2,G3,True_Label,Pred_Label,Confidence,Prob_Class_0,Prob_Class_1,Prob_Class_2
    ```
  * In `src/evaluation.py` lines 123-131, predictions are also persisted into the PostgreSQL database under the table `paper_predictions`.

* **MLP Model Structures**:
  * **Context MLP** (inside `StudentHybridModel` in `src/models.py` lines 102-108):
    ```python
            context_input_dim = num_numerical + embedding_total_dim
            self.context_input_dim = max(1, context_input_dim)
            self.context_mlp = nn.Sequential(
                nn.Linear(self.context_input_dim, context_hidden_dim),
                nn.ReLU(),
                nn.Dropout(context_dropout),
                nn.Linear(context_hidden_dim, context_hidden_dim),
                nn.ReLU(),
            )
    ```
  * **Recommendation MLP** (in `src/recommendation.py` lines 138-152):
    ```python
    class RecommendationMLP(nn.Module):
        def __init__(self, input_dim: int, output_dim: int = 6):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(input_dim, 64),
                nn.ReLU(),
                nn.Dropout(0.10),
                nn.Linear(64, 32),
                nn.ReLU(),
                nn.Linear(32, output_dim),
            )

        def forward(self, features: torch.Tensor) -> torch.Tensor:
            return self.net(features)
    ```
  * **Recommendation MLP** (older model in `src/explainability.py` lines 149-161):
    ```python
    class RecommendationMLP(torch.nn.Module):
        def __init__(self, input_dim: int, output_dim: int = 6):
            super().__init__()
            self.net = torch.nn.Sequential(
                torch.nn.Linear(input_dim, 64),
                torch.nn.ReLU(),
                torch.nn.Linear(64, 32),
                torch.nn.ReLU(),
                torch.nn.Linear(32, output_dim)
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.net(x)
    ```

* **Recommendation and Evaluation Logic**:
  * **Recommendation**:
    * Production logic is implemented in `MLPLearningPathEngine` in `src/recommendation.py`.
    * Features are extracted from students using `extract_features` (8 features for student kind, 7 features for xapi kind).
    * Features are normalized using `(features - mean) / scale` stored in the checkpoint.
    * Model outputs are converted to probabilities using `torch.sigmoid`.
    * Active risks are chosen where `score >= 0.5`. If no risk meets this criteria and the predicted class is not 2 (High), it falls back to the risk with the highest probability.
    * Risk factors are sorted by priority ascending, then by score descending: `(item.priority, -item.score)`.
    * It maps risk factors to staged weekly recommendations (phases, goals, actions) using `_student_actions` or `_xapi_actions`.
  * **Evaluation**:
    * Implemented in `src/eval_recommendation.py`.
    * Computes precision_macro, recall_macro, f1_macro, and hamming loss for multi-label predictions thresholded at 0.5 against domain reference targets (`reference_risk_targets`).
    * Computes ranking metrics: precision_at_k, recall_at_k, and ndcg_at_k for K in (1, 3, 5).
    * Computes structural quality metrics: `nonempty_path_rate`, `complete_step_schema_rate`, `staged_path_rate`.
    * Leaves LLM-Judge as `"not_run"` / `None`.

* **Model Checkpoints**:
  * Located under:
    * `models/recommendation/student-mat_mlp.pt`
    * `models/recommendation/student-por_mlp.pt`
    * `models/recommendation/xapi_mlp.pt`
  * Old model weights:
    * `models/mlp_rec_student.pt`
    * `models/mlp_rec_xapi.pt`
  * Performance predictor ensemble weights (11 seeds):
    * `models/saved/final/*_3class_cnn_bilstm_mlp_seed*.pt`
  * Pre-trained checkpoints loaded in `models/`:
    * `models/best_student-mat.pt`
    * `models/best_student-por.pt`
    * `models/best_xapi.pt`

* **Locked Metrics**:
  * Located under `reports/final/metrics/`:
    * `student-mat_3class_locked_test_metrics.json`
    * `student-por_3class_locked_test_metrics.json`
    * `xapi_3class_locked_test_metrics.json`
    * (and equivalent `*_v27_locked_test_metrics.json` files).
  * Example (`student-mat_3class_locked_test_metrics.json` content):
    ```json
    {
        "Accuracy": 0.8607594936708861,
        "F1-Macro": 0.8689935900435128,
        "Precision-Macro": 0.8600335249042147,
        "Recall-Macro": 0.899460188933873,
        "RMSE": 0.3731494423540171,
        "R2": 0.7212957023733162
    }
    ```

* **Verification Task Output**:
  * Ran test command: `C:\Users\THPhu\anaconda3\envs\kltn\python.exe -m pytest -q`
  * Test execution failed with:
    ```
    FAILED tests/test_thesis_pipeline.py::test_forbidden_architectures_and_losses_are_removed
    1 failed, 9 passed in 11.55s
    ```
  * Specifically, the assertion `assert forbidden not in source` failed for `"FocalLoss"` because `FocalLoss` is defined in `src/models.py` at line 11.

## 2. Logic Chain
1. We searched for CSV writing calls (`to_csv`) and verified the path definitions in `src/config.py` and `scripts/run_pipeline.py`, which led us directly to `reports/final/predictions/` and database schema tables.
2. We inspected `src/models.py`, `src/recommendation.py`, and `src/explainability.py` files to extract the exact sequential structures of the MLP models used in both context processing and recommendation.
3. We traced the method `generate()` and `predict_scores()` in `MLPLearningPathEngine` and `RuleBasedLearningPathEngine` to compare feature extraction, normalization, sorting, and action templates.
4. We verified `src/eval_recommendation.py` to identify the multilabel, ranking, and structural metrics evaluated on the locked-test splits.
5. We searched for files matching `.pt` and `torch.load` to establish a catalog of model checkpoints across folders.
6. We located the JSON files containing the locked metrics under `reports/final/metrics/` and retrieved their exact values.
7. We attempted to verify the codebase testing pipeline using the `kltn` conda environment, discovering a test failure caused by the presence of `FocalLoss` in `src/models.py`.

## 3. Caveats
- The PostgreSQL database backend was not inspected directly as connection credentials were not validated, but we confirmed schema mapping in `src/evaluation.py` and `database/schema.sql`.
- The test failure of `test_forbidden_architectures_and_losses_are_removed` is documented but left unfixed due to the read-only constraint of this task.

## 4. Conclusion
- The outputs of the CNN-BiLSTM performance predictor are saved in `reports/final/predictions/` and persisted to database table `paper_predictions`.
- The current MLP models consist of a Context MLP inside the hybrid model and a separate `RecommendationMLP` for multilabel risk prediction.
- Recommendation logic leverages `MLPLearningPathEngine` (with normalization and priority-based sorting) and is evaluated via `src/eval_recommendation.py`.
- Locked baseline metrics reside in `reports/final/metrics/` and checkpoints in `models/recommendation/`.
- Downstream tasks should leave these baseline files untouched but must address the failing test where `FocalLoss` presence contradicts project architecture rules.

## 5. Verification Method
- Execute the test suite using:
  ```powershell
  C:\Users\THPhu\anaconda3\envs\kltn\python.exe -m pytest
  ```
- Inspect the file locations described in this report:
  * `reports/final/predictions/`
  * `reports/final/metrics/`
  * `models/recommendation/`
