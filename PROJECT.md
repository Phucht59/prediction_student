# Project: Rebuilding Recommendation Model with PyTorch MLP

## Architecture
- Code base uses CNN-BiLSTM + MLP for student performance prediction (in `src/models.py`).
- The current recommendation system uses `RuleBasedLearningPathEngine` in `src/explainability.py` to map academic risks to staged roadmaps.
- We need to replace it with a PyTorch MLP recommendation model.
- We need to write a new script `src/eval_recommendation.py` for evaluating recommendations (NDCG, Precision, Recall, LLM-Judge).

## Milestones
| # | Name | Scope | Dependencies | Status |
|---|------|-------|-------------|--------|
| 1 | Exploration | Explore codebase, examine data, understand current engine and how recommendations are generated and tested | None | DONE |
| 2 | Rebuild Rec Model | Implement the train-pool-only PyTorch MLP risk ranker in `src/recommendation.py` | M1 | DONE |
| 3 | Evaluation Pipeline | Compute held-out Precision@K, Recall@K and NDCG@K; leave LLM-Judge unset without independent ratings | M2 | DONE |
| 4 | Verification & Audit | Run tests, three 11-seed ensembles, fixed 5-fold CV, DOCX render QA and artifact checks | M3 | DONE |

## Interface Contracts
### Recommendation Engine interface
- `MLPLearningPathEngine` is the production recommendation interface.
- Method `generate(features: dict, predicted_class: int, confidence: float) -> dict` must be preserved or compatible with existing code to avoid breaking existing pipeline integrations.
- Recommendations must return structured learning path containing phases, goals, actions, and list of risk factors.
