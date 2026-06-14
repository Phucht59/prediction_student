# Handoff Report

## 1. Observation
- Observed `FocalLoss` definition in `src/models.py` and its import/usage in `src/train_pipeline.py` and `scripts/run_pipeline.py`.
- Observed the rule-based logic to extract risks in `_student_risks` and `_xapi_risks` inside `src/explainability.py`.
- Checked the `tests/test_thesis_pipeline.py` which contains tests verifying that FocalLoss is absent from models.py, and that the engine works properly.
- Ran command `C:\Users\THPhu\anaconda3\envs\kltn\python.exe -m pytest -v` before any changes, which passed except for the `test_forbidden_architectures_and_losses_are_removed` (since FocalLoss was defined in `src/models.py`).
- Ran command `C:\Users\THPhu\anaconda3\envs\kltn\python.exe src/eval_recommendation.py` which completed successfully and generated the evaluation JSON reports under `reports/final/recommendations/`.
- Ran command `C:\Users\THPhu\anaconda3\envs\kltn\python.exe -m pytest -v` after changes, which passed all 10 tests.

## 2. Logic Chain
- Deleting the `FocalLoss` class entirely from `src/models.py` and updating the references in `src/train_pipeline.py` and `scripts/run_pipeline.py` to use weighted `nn.CrossEntropyLoss` addresses the architecture violation and resolves `test_forbidden_architectures_and_losses_are_removed`.
- Implementing `extract_student_features` and `extract_xapi_features` maps the structured features (both raw strings and numbers) into tensors suitable for feed-forward neural networks.
- Implementing the `RecommendationMLP` model class and training it for 150 epochs of `BCEWithLogitsLoss` using the Adam optimizer maps student feature vectors to the 6 risk factors.
- Modifying the `RuleBasedLearningPathEngine` class's `__init__` and `generate` methods to check, auto-train (if weights are missing), load, and predict risks using the trained MLP matches the PyTorch model requirement.
- Creating the script `src/eval_recommendation.py` to load the locked test sets, compute Precision/Recall/NDCG@K (where K=1, 3, 5), run LLM/NLP judge evaluation, and export the average metrics to JSON matches the requested evaluation pipeline requirements.

## 3. Caveats
- It is assumed that the raw datasets are located under `data/raw/` in the format `student-mat.csv`, `student-por.csv`, and `xAPI-Edu-Data.csv`. If these files are missing or modified, the self-training routine will fail.

## 4. Conclusion
- The FocalLoss architecture mismatch has been fixed, the recommendation model has been successfully rebuilt using PyTorch MLP with self-training and inference logic, and the evaluation pipeline has been fully implemented and verified.

## 5. Verification Method
To verify the changes:
1. Run pytest to ensure all unit tests pass:
   ```bash
   C:\Users\THPhu\anaconda3\envs\kltn\python.exe -m pytest -v
   ```
2. Run the evaluation script:
   ```bash
   C:\Users\THPhu\anaconda3\envs\kltn\python.exe src/eval_recommendation.py
   ```
3. Verify that the three JSON reports are correctly created under `reports/final/recommendations/`:
   - `student_mat_evaluation.json`
   - `student_por_evaluation.json`
   - `xapi_evaluation.json`
