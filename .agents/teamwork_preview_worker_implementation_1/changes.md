# Changes Made

1. **Remove FocalLoss & Standardize Criterion**:
   - `src/models.py`: Removed the `FocalLoss` class completely.
   - `src/train_pipeline.py`: Removed imports of `FocalLoss` and replaced the focal loss logic with weighted `nn.CrossEntropyLoss` for optimizing student models.
   - `scripts/run_pipeline.py`: Removed imports of `FocalLoss` and replaced it with weighted `nn.CrossEntropyLoss` when optimizing/training student models.

2. **Rebuild Recommendation Model using PyTorch MLP**:
   - `src/explainability.py`:
     - Added `extract_student_features` and `extract_xapi_features` helper functions to map student/xapi data dictionary features to MLP input floats.
     - Added `RecommendationMLP` model class (3-layer MLP) mapping 8 (student) or 7 (xapi) inputs to 6 output logits (representing the 6 risk factors).
     - Added automated training routine (`_train_mlp`) during initialization of `RuleBasedLearningPathEngine` which fits the MLP model on raw data (150 epochs of `BCEWithLogitsLoss`) if the weights do not exist, saving them to `models/mlp_rec_student.pt` or `models/mlp_rec_xapi.pt`.
     - Updated `generate()` method to run the MLP forward pass, apply `sigmoid`, threshold at `> 0.5` to find active risk factors, and construct the correct recommendation roadmap.

3. **Build Recommendation Evaluation Pipeline**:
   - `src/eval_recommendation.py`: Created script to load the locked test sets (`student-mat_3class_locked_test.csv`, `student-por_3class_locked_test.csv`, `xapi_3class_locked_test.csv`). For each student, it compares predicted risks against the ground-truth rules, calculates Precision@K, Recall@K, NDCG@K (K=1, 3, 5), runs the LLM / local fallback judge (evaluating alignment, feasibility, progression, and overall metrics), and saves the results as JSON.

4. **Verify correctness**:
   - Run unit tests to verify that forbidden architectures/losses test (`test_forbidden_architectures_and_losses_are_removed`) and engine tests pass.
   - Run the recommendation evaluation pipeline to verify JSON outputs.
