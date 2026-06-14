# Plan: Rebuilding Recommendation Model with PyTorch MLP

## Objective
Rebuild the Recommendation Model using PyTorch MLP, implement the Evaluation Pipeline in `src/eval_recommendation.py`, verify performance, and satisfy all constraints.

## Milestones
- **Milestone 1: Exploration & Setup** [DONE]
  - Spawn an Explorer agent to explore the directory structure, existing tests, data pipeline, and training script.
  - Determine where dataset matrices or features are located.
  - Run current tests to establish a baseline.
- **Milestone 2: Rebuild Recommendation Model (PyTorch MLP)** [DONE]
  - Replace `RuleBasedLearningPathEngine` with an MLP-based model.
  - Define model architecture and weight/training scheme for recommendations.
  - Ensure compatibility with existing calls in `src/explainability.py` and `tests/test_thesis_pipeline.py`.
- **Milestone 3: Implement Evaluation Pipeline** [DONE]
  - Build `src/eval_recommendation.py` to evaluate recommendations quantitatively (Precision@K, Recall@K, NDCG@K) and qualitatively (LLM-Judge).
  - Save JSON results to `reports/final/recommendations/`.
- **Milestone 4: Verification, Review & Audit** [DONE]
  - Perform unit, E2E, and adversarial testing.
  - Validate with a Forensic Auditor to guarantee no changes were made to resampling/preprocessing logic.
