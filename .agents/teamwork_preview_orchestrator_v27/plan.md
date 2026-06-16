# V27 Model Improvement Project Plan

## Architecture Overview
- Codebase uses CNN-BiLSTM + MLP for student performance prediction (in `src/models/models.py`).
- We need to improve prediction of this architecture across 3 datasets: `student-mat`, `student-por`, `xapi`.
- Model architecture V27: `StudentHybridV27` with a sequence branch (Conv1D + BiLSTM) and a context branch (Embeddings + Context MLP), fused via Gated Fusion, plus auxiliary heads (Ordinal + Regression).

## Milestones & Verification Plan

### Milestone 1: Exploration & Pipeline Audit
- **Objective**: Audit the current training pipeline and data pipeline. Identify data leakage, evaluate current resampling logic, and prepare the design for SMOTENC/ADASYN and V27 architecture.
- **Verification**: Explorer handoff report with analysis of current resampling, train/val/test splitting, and a safe data preparation protocol.

### Milestone 2: Resampling Fix and Loss/Architecture Implementation
- **Objective**: Implement safe resampling (SMOTENC for mixed data, ADASYN only for numeric-safe if used; never on val/test) and write `src/models_v27.py` (`StudentHybridV27`) and `src/losses_v27.py`.
- **Verification**: Clean build, unit tests for models/losses passing, and comparison table `outputs/experiments/resampling_comparison.csv` generated.

### Milestone 3: Optuna Hyperparameter & Threshold Tuning
- **Objective**: Tune models using Optuna on validation sets (not locked test), tune classification decision thresholds on validation, and save thresholds.
- **Verification**: Scripts running without errors; hyperparameter trials logged, thresholds saved to `outputs/experiments/thresholds_{dataset}.json`.

### Milestone 4: Seed Ensembling & Ablation Study
- **Objective**: Implement and run seed ensembling (seeds 42-46) and the ablation study with 10 variants.
- **Verification**: Artifacts `outputs/v27/{dataset}/ensemble_metrics.json` and `outputs/v27/ablation_results.csv` successfully generated.

### Milestone 5: Evaluation & Final Reporting
- **Objective**: Run final evaluation on the locked test sets, check performance targets against baseline, and generate the final report file.
- **Verification**: Final predictions and metrics saved, `outputs/v27/final_prediction_section.md` created with actual results.
