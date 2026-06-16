# BRIEFING — 2026-06-15T15:06:00+07:00

## Mission
Explore the data pipeline, model architectures, and resampling strategies, auditing for leaks and designing StudentHybridV27 & loss functions.

## 🔒 My Identity
- Archetype: Codebase Explorer and Pipeline Auditor
- Roles: Codebase Explorer, Pipeline Auditor
- Working directory: c:\Huflit\kltn\.agents\teamwork_preview_explorer_exploration_v27_1
- Original parent: 2d42b4cb-2222-43ba-9436-ae0707b291c0
- Milestone: exploration_v27_1

## 🔒 Key Constraints
- Read-only investigation — do NOT implement
- Code-only network mode (no external access, no HTTP client calls, use code_search/filesystem search tools and view_file)

## Current Parent
- Conversation ID: 2d42b4cb-2222-43ba-9436-ae0707b291c0
- Updated: 2026-06-15T15:06:00+07:00

## Investigation State
- **Explored paths**:
  - `src/data_pipeline.py` (inspected target prep, preprocessing, feature engineering, feature selector, dataset loader)
  - `src/train_pipeline.py` (inspected training epochs, Optuna objective function, parameter suggestions)
  - `scripts/run_pipeline.py` (inspected split creation, ensemble training, training loop structure)
  - `src/models/models.py` (inspected model architecture, forward pass, categorical embeddings, attention pooling)
- **Key findings**:
  - Preprocessing fits scalers and encoders on training folds only, preventing direct validation/test leakage.
  - Split-before-resample is correctly enforced in `train_seed_ensemble`.
  - Feature selection is applied *after* resampling inside Optuna objective and ensemble loops, which is a methodological leak/distortion (computes statistics on synthetic data).
  - ADASYN is selected for the student dataset but behaves incorrectly with categorical features. It outputs floats for categorical variables, which are then brutally truncated to ints, causing representation distortion.
  - The model targets are binned into classes before writing to the CSVs, losing raw continuous targets needed for auxiliary regression heads.
- **Unexplored areas**: None, codebase audit complete.

## Key Decisions Made
- Confirmed validation splitting mechanism is safe from direct leakages.
- Identified ADASYN categorical handling and feature selection order as major issues.
- Designed Gated Fusion and auxiliary heads for StudentHybridV27 model.
- Designed losses including Class-Balanced Focal and Ordinal losses.

## Artifact Index
- c:\Huflit\kltn\.agents\teamwork_preview_explorer_exploration_v27_1\handoff.md — Comprehensive exploration findings report
