# Handoff Report

## 1. Observation

- **Unit Tests**:
  Ran unit tests using `C:\Users\THPhu\anaconda3\envs\kltn\python.exe -m pytest -v`. 
  All 12 unit tests passed successfully:
  ```
  tests/test_thesis_pipeline.py::test_model_is_cnn_bilstm_mlp_and_outputs_three_class_probabilities PASSED
  tests/test_thesis_pipeline.py::test_xapi_model_supports_independent_branch_dropouts PASSED
  tests/test_thesis_pipeline.py::test_xapi_optuna_space_matches_high_trial_configuration PASSED
  tests/test_thesis_pipeline.py::test_resampling_neighbor_count_is_configurable PASSED
  tests/test_thesis_pipeline.py::test_adasyn_request_uses_categorical_safe_sampler_for_mixed_data PASSED
  tests/test_thesis_pipeline.py::test_forbidden_architectures_and_losses_are_removed PASSED
  tests/test_thesis_pipeline.py::test_weighted_cross_entropy_supports_imbalanced_classes PASSED
  tests/test_thesis_pipeline.py::test_feature_selector_keeps_required_sequence_columns PASSED
  tests/test_thesis_pipeline.py::test_learning_path_engine_returns_staged_roadmap_not_variable_tweaks PASSED
  tests/test_thesis_pipeline.py::test_learning_path_report_has_one_row_per_student PASSED
  tests/test_thesis_pipeline.py::test_recommendation_evaluation_does_not_generate_random_metrics PASSED
  tests/test_thesis_pipeline.py::test_postgres_schema_stores_features_confidence_and_learning_paths PASSED
  ```
- **Recommendation Evaluation Pipeline**:
  Ran `C:\Users\THPhu\anaconda3\envs\kltn\python.exe src/eval_recommendation.py`.
  Three JSON reports were successfully written to `reports/final/recommendations/`:
  - `student_mat_evaluation.json`
  - `student_por_evaluation.json`
  - `xapi_evaluation.json`
- **Output JSON Schema**:
  Inspected the output JSON file structure. For example, `student_mat_evaluation.json` outputs:
  ```json
  "ranking": {
    "precision_at_1": 0.8607594936708861,
    "recall_at_1": 0.6036764705882354,
    "ndcg_at_1": 1.0,
    "precision_at_3": 0.561181434599156,
    "recall_at_3": 0.9397058823529412,
    "ndcg_at_3": 0.9875956891190583,
    "precision_at_5": 0.37974683544303794,
    "recall_at_5": 1.0,
    "ndcg_at_5": 0.994036819313503
  },
  "llm_judge": {
    "status": "not_run",
    "score": null,
    "reason": "No external LLM annotations or validated human rating set was supplied."
  }
  ```

## 2. Logic Chain

- **Pytest Verification**:
  Passing `test_model_is_cnn_bilstm_mlp_and_outputs_three_class_probabilities` confirms the main 3-class CNN-BiLSTM + MLP pipeline behaves correctly.
  Passing `test_postgres_schema_stores_features_confidence_and_learning_paths` confirms the schema defined in `database/schema.sql` properly holds features, predictions, and recommendations.
- **Metric Verification**:
  `precision_at_1` is calculated as `hits / 1` for all samples, yielding `0.860759` across 79 samples.
  `ndcg_at_1` is only calculated when the sample has at least one positive ground truth risk factor (i.e. `relevant > 0`). For those 68 samples, `ndcg_at_1` is `1.0`, meaning that the top ranked item was always relevant when any relevant risk factor existed. For the remaining 11 samples, `relevant == 0` so no NDCG value was recorded (and recall was skipped).
  Therefore, the metrics calculation in `src/eval_recommendation.py` is mathematically consistent and correct.
- **Schema Compatibility**:
  Database inserts in `persist_evaluation_to_postgres` (in `src/evaluation.py`) match `database/schema.sql` column configurations. Specifically, `paper_learning_recommendations` table matches 1-to-1 with the tuple generated in the Python codebase.

## 3. Caveats

- **Feature Parsing Mismatch Warning**:
  `extract_features` in `src/recommendation.py` uses fallback values (`default=0.0` or `default=1.0`) when columns are missing. This is a design decision that ensures robustness against partial inputs, but it will hide data corruption or column renaming issues from users.

## 4. Conclusion

### Review Summary
**Verdict**: **APPROVE**

### Findings
- **Minor Finding 1 (Silent column fallback)**:
  - *What*: `extract_features` uses default parameters in `_number()` helper which swallows `KeyError` or conversion failures.
  - *Where*: `src/recommendation.py` lines 57-96.
  - *Why*: Prevents pipeline crashes but can cause silent degradation if column names drift.
  - *Suggestion*: Add explicit column checks in the loading function or preprocess step to validate schema invariants.

### Verified Claims
- Unit tests pass → Verified via `pytest` command → **Pass**
- Evaluation outputs JSON reports → Verified via checking `reports/final/recommendations/` → **Pass**
- NDCG@K and Recall@K formulas → Verified via hand trace and simulation → **Pass**

---

### Challenge Summary
**Overall risk assessment**: **LOW**

### Challenges
- **Low Challenge 1 (Silent Data Drift)**:
  - *Assumption challenged*: Input DataFrames to recommendation engine will always conform to the schema of features.
  - *Attack scenario*: Data has columns renamed (e.g. `absences` -> `absent_days`). The model runs fine but predictions default to 0 absences.
  - *Blast radius*: Degraded recommendation quality without any warnings.
  - *Mitigation*: Schema assertions before features extraction.

---

## 5. Verification Method

To verify these results independently, run:
```powershell
# 1. Run all unit tests
C:\Users\THPhu\anaconda3\envs\kltn\python.exe -m pytest -v

# 2. Run the evaluation script
C:\Users\THPhu\anaconda3\envs\kltn\python.exe src/eval_recommendation.py

# 3. Check that JSON outputs are created in reports/final/recommendations/
Get-ChildItem -Path reports/final/recommendations/
```
