# Project: V27 Model Improvement

## Architecture
- Code base uses CNN-BiLSTM + MLP for student performance prediction (in `src/models/models.py`).
- We need to improve prediction of this architecture across 3 datasets: `student-mat`, `student-por`, `xapi`.
- Model architecture V27: `StudentHybridV27` with sequence branch (Conv1D + BiLSTM) and context branch (Embeddings + Context MLP) fused via Gated Fusion, plus auxiliary heads (Ordinal + Regression).

## Milestones
| # | Name | Scope | Dependencies | Status |
|---|------|-------|-------------|--------|
| 1 | Exploration & Pipeline Audit | Explore codebase, check data leakage, evaluate current resampling logic. | None | DONE (0db65ef1-b3b0-45b8-a2e5-4e10daefb216) |
| 2 | Resampling & V27 Model Implementation | Implement SMOTENC, `StudentHybridV27` model, loss functions, and run resampling baseline comparisons. | M1 | DONE |
| 3 | Optuna & Threshold Tuning | Run Optuna search for hyperparameters and tune prediction thresholds on validation set. | M2 | DONE |
| 4 | Seed Ensembling & Ablation Study | Train seed ensembles (seeds 42-46) and evaluate 10 ablation variants. | M3 | DONE |
| 5 | Evaluation & Final Reporting | Final evaluation on locked test, compare with baselines, write `outputs/v27/final_prediction_section.md`. | M4 | DONE |

## Interface Contracts
### Model V27 Interface (`src/models_v27.py`)
- Class: `StudentHybridV27(nn.Module)`
- Inputs: sequence inputs, categorical context inputs, numerical context inputs.
- Outputs: main performance class predictions, ordinal class probabilities, regression score (if applicable).
- Target files: `src/models_v27.py`, `src/losses_v27.py`, `scripts/run_v27_optuna.py`, `scripts/run_v27_pipeline.py`.
