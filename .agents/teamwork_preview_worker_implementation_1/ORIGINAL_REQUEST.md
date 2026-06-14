## 2026-06-14T12:19:21Z
You are the Recommendation Developer.
Your task is to implement the following changes in the codebase:
1. Fix the FocalLoss architecture mismatch:
   - In `src/models.py`, remove the `FocalLoss` class entirely.
   - In `src/train_pipeline.py`, remove references to `FocalLoss` (including imports and usages), replacing it with weighted `nn.CrossEntropyLoss` when optimizing.
2. Rebuild the Recommendation Model using a PyTorch MLP:
   - In `src/explainability.py`, define a PyTorch MLP model class for mapping features to risks (e.g. inputs size 8 for student, 7 for xapi, outputs size 6 for the 6 risk factors).
   - Implement an automated self-training routine: when `RuleBasedLearningPathEngine` is initialized, if the model weights (e.g., `models/mlp_rec_student.pt` or `models/mlp_rec_xapi.pt`) do not exist, load the raw datasets (using pandas), generate the ground-truth binary targets using the rules, train the MLP model on the features to predict the 6 risk factors (e.g., 150 epochs of BCEWithLogitsLoss), and save the weights.
   - Modify the `generate()` method of `RuleBasedLearningPathEngine` to run a forward pass of the trained PyTorch MLP model on the student's features, obtain the predicted risk probabilities, threshold them (> 0.5) to determine active risk factors, and then generate the stages, goals, and actions matching the output structure of the original engine.
3. Build the Evaluation Pipeline:
   - Create the script `src/eval_recommendation.py`.
   - The script should load the locked test sets (`student-mat_3class_locked_test.csv`, `student-por_3class_locked_test.csv`, `xapi_3class_locked_test.csv`) from `data/processed/final/`.
   - For each student, compute the ground-truth active risks using the rules and the predicted active risks using the MLP model.
   - Calculate Precision@K, Recall@K, and NDCG@K (where K=1, 3, 5) for the retrieved risk factors.
   - Implement an automated LLM-Judge scorer according to pedagogical criteria (alignment, feasibility, progression, overall). Check for API keys (e.g., `OPENAI_API_KEY`, `GEMINI_API_KEY`) and use them if available; if not or if the call fails, implement a robust local NLP keyword/regex check as a fallback that verifies alignment between features and actions, returning structured quality grades.
   - Run this evaluation over the locked test sets, compute average metrics, and save the results as JSON to `reports/final/recommendations/` (e.g., `student_mat_evaluation.json`, `student_por_evaluation.json`, `xapi_evaluation.json`).
4. Verification:
   - Run the test suite using `python -m pytest -v` (using the environment at `C:\Users\THPhu\anaconda3\envs\kltn`) to ensure all tests pass (including the fixed forbidden architectures test).
   - Test the evaluation script end-to-end to ensure it creates the JSON reports correctly.

MANDATORY INTEGRITY WARNING:
> DO NOT CHEAT. All implementations must be genuine. DO NOT
> hardcode test results, create dummy/facade implementations, or
> circumvent the intended task. A Forensic Auditor will independently
> verify your work. Integrity violations WILL be detected and your
> work WILL be rejected.

Your working directory is: c:\Huflit\kltn\.agents\teamwork_preview_worker_implementation_1
Please keep track of your progress in `progress.md` and document your changes in `changes.md` and `handoff.md` in your working directory.
Your parent is: Project Orchestrator (conversation ID: 5ec1de11-4fc2-4756-80ed-d011dd7a9b96). Report your results and handoff back to this conversation ID.
