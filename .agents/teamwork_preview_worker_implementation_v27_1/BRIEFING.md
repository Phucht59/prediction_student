# BRIEFING — 2026-06-15T08:19:30Z

## Mission
Implement data pipeline fixes, new StudentHybridV27 model architecture, loss functions, V27 training pipeline, and run resampling experiments.

## 🔒 My Identity
- Archetype: Machine Learning and Architecture Implementer
- Roles: implementer, qa, specialist
- Working directory: c:\Huflit\kltn\.agents\teamwork_preview_worker_implementation_v27_1
- Original parent: 369625da-5db3-49c8-9991-d298107f902b
- Milestone: v27_1

## 🔒 Key Constraints
- CODE_ONLY network mode: no external requests, only local code searches and modifications.
- Minimal change principle.
- Use explicit editing tools.
- Never write project source code in `.agents/`.

## Current Parent
- Conversation ID: 369625da-5db3-49c8-9991-d298107f902b
- Updated: 2026-06-15T08:19:30Z

## Task Summary
- **What to build**: 
  1. Resampling and Pipeline Fixes: Force SMOTENC if categorical features exist, round/cast resampled categoricals. Save/drop G3_raw, update StudentDataset, ensure feature selection before oversampling.
  2. Model Architecture: src/models_v27.py containing AttentionPooling1D, GatedFusion, StudentHybridV27.
  3. Loss Functions: src/losses_v27.py containing FocalLoss, ClassBalancedFocalLoss, OrdinalLoss, JointHybridLoss.
  4. Training Pipeline: src/train_v27_pipeline.py and scripts/run_v27_pipeline.py.
  5. Resampling Comparison: Experiment script comparing None, SMOTE, SMOTENC, ADASYN on student-mat and student-por.
- **Success criteria**: All tests pass, metrics are generated, code is well-structured and conforms to guidelines.
- **Interface contracts**: src/data_pipeline.py, src/models_v27.py, etc.
- **Code layout**: src/ for modules, scripts/ for executable entry points, tests/ for testing.

## Key Decisions Made
- Exclude `G3_raw` from the list of features inside both `DataPreprocessor` and `FeatureSelector` to prevent data leakage, but retain it in the DataFrame columns to allow `StudentDataset` to retrieve it.
- Dynamically determine whether categorical columns are present during oversampling, and if so, instantiate `SMOTENC` to prevent floats being generated for categoricals, resolving the ADASYN/SMOTENC bug.
- Slice dataloader batches to `batch[:5]` in legacy loops for backward compatibility.
- Use a subprocess wrapper for running comparison experiments to cleanly capture and monitor output.

## Artifact Index
- `src/data_pipeline.py` — Updated preprocessor and dataset classes to handle G3_raw and correct SMOTENC resampling.
- `src/models_v27.py` — Custom V27 student hybrid model with AttentionPooling, GatedFusion, and three heads.
- `src/losses_v27.py` — Focal, Class-Balanced, Ordinal, and JointHybrid loss classes.
- `src/train_v27_pipeline.py` — V27 model training logic and helpers.
- `scripts/run_v27_pipeline.py` — Runs fixed 5-fold cross-validation on V27 models.
- `scripts/compare_resampling.py` — Resampling comparison experiment script.
- `tests/test_v27_components.py` — Unit test suite verifying all V27 elements.
- `outputs/v27/` — Folders containing generated metrics JSON files.
- `outputs/experiments/resampling_comparison.csv` — Results of resampling methods comparison.
