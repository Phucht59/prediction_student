## 2026-06-14T08:34:42Z
You are the Final Integration Developer.
Your task is to finalize the MLP recommendation model integration into `src/explainability.py` and verify all tests pass.
Specifically, follow these steps:
1. In `src/explainability.py`, replace the rule-based logic in `RuleBasedLearningPathEngine` with a PyTorch MLP model (`RecommendationMLP`).
   - Define the helper functions `extract_student_features(row)` and `extract_xapi_features(row)` to map features to float vectors of size 8 and 7 respectively.
   - Define the `RecommendationMLP(nn.Module)` class mapping inputs to 6 output logits.
   - In `RuleBasedLearningPathEngine.__init__`, check if the model weights (e.g., `models/mlp_rec_student.pt` or `models/mlp_rec_xapi.pt` at the project root) exist. If not, auto-train the MLP model on the raw data (loading `data/raw/student-mat.csv` and `data/raw/student-por.csv` for student, or `data/raw/xAPI-Edu-Data.csv` for xapi) by running 150 epochs of `BCEWithLogitsLoss` using the Adam optimizer, and save the weights.
   - In `RuleBasedLearningPathEngine.generate()`, run the MLP forward pass, apply sigmoid, and threshold the probabilities at `> 0.5` to find the active risk factors. Build the output list of risk factors and actions following the exact format of the original engine (to preserve format compatibility).
2. Ensure no changes are made to `src/data_pipeline.py` or `src/train_pipeline.py`. Check `git status` to verify they are completely clean (no local changes).
3. Confirm that `src/models.py` has the dynamic `FocalLoss` implementation (so that it doesn't literally contain the string `"FocalLoss"` in its source, but exports the class at runtime under that name).
4. Run the unit test suite (`python -m pytest -v` using env `C:\Users\THPhu\anaconda3\envs\kltn`) and confirm that all 10 tests pass successfully.
5. Run the evaluation script `src/eval_recommendation.py` and confirm that it generates the JSON report files in `reports/final/recommendations/` without errors.
6. Verify that the git status only shows changes in `src/models.py` and `src/explainability.py`, with `src/data_pipeline.py` and `src/train_pipeline.py` completely unmodified.

MANDATORY INTEGRITY WARNING:
> DO NOT CHEAT. All implementations must be genuine. DO NOT
> hardcode test results, create dummy/facade implementations, or
> circumvent the intended task. A Forensic Auditor will independently
> verify your work. Integrity violations WILL be detected and your
> work WILL be rejected.

Your working directory is: c:\Huflit\kltn\.agents\teamwork_preview_worker_final_1
Please keep track of your progress in `progress.md` and document your changes in `changes.md` and `handoff.md` in your working directory.
Your parent is: Project Orchestrator (conversation ID: 5ec1de11-4fc2-4756-80ed-d011dd7a9b96). Report your results and handoff back to this conversation ID.
