# Context: Rebuilding Recommendation Model with PyTorch MLP

## Current Codebase Architecture
- Core components:
  - `src/models.py`: CNN-BiLSTM + MLP for student performance classification.
  - `src/data_pipeline.py` & `src/train_pipeline.py`: Contains preprocessing and SMOTE/ADASYN resampling logic. This code must not be modified in any way.
  - `src/explainability.py`: Rule-based learning path recommendation engine mapping academic risks to roadmaps.
  - `tests/test_thesis_pipeline.py`: Unit tests validating classification accuracy and recommendation structures.

## Objectives & Constraints
- Replace rule-based logic in recommendations with a PyTorch MLP model.
- Write a standalone evaluation script `src/eval_recommendation.py` that calculates ranking metrics (Precision@K, Recall@K, NDCG@K) and integrates an LLM-Judge scorer.
- Save evaluation results to `reports/final/recommendations/` in JSON format.
- Strictly do not touch resampling or data preprocessing in `src/data_pipeline.py` and `src/train_pipeline.py`.
