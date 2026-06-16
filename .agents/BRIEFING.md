# BRIEFING — 2026-06-15T08:54:00Z

## Mission
Improve the student performance prediction model (CNN-BiLSTM + Context MLP) V27 on three datasets (student-mat, student-por, xapi) to optimize Macro-F1 and Recall Low while ensuring strict data leakage protection.

## 🔒 My Identity
- Archetype: sentinel
- Working directory: c:\Huflit\kltn\.agents
- Orchestrator: 2d42b4cb-2222-43ba-9436-ae0707b291c0
- Victory Auditor: f6c96679-8f29-4ed3-9253-f582b436a4aa

## 🔒 Key Constraints
- No technical decisions — relay only
- Victory Audit is MANDATORY before reporting completion
- Keep pre-processing and resampling logic strictly unchanged in src/data_pipeline.py and src/train_pipeline.py (except as requested for the V27 pipeline specifically)
- Do not modify or break the existing CNN+BiLSTM prediction pipeline or test metrics
- Recommend system must operate completely as a downstream module
- Refactor PyTorch MLP model only as Risk Diagnosis Head, not the main recommender
- Intervention catalog must contain realistic education actions
- No metric fabrication. Do not report evaluation table for datasets that haven't been run.
- Do not call it collaborative filtering if there's no user-item interaction data.
- Do not call it knowledge graph if no actual graph is constructed.
- Do not use risks without corresponding features in the dataset.
- Audit & Fix Resampling: ensure locked test is completely isolated (no data leakage).
- Use SMOTENC or ADASYN (only for numeric-safe) on training set, never on validation or locked test.
- Implement StudentHybridV27 (CNN-BiLSTM + Context MLP with Gated Fusion, Ordinal/Regression auxiliary heads).
- Test various loss functions (Weighted CE, Focal Loss, CB-Focal, Ordinal loss) and output loss comparison.
- Run Optuna tuning on train/val, not on locked test.
- Tune decision thresholds on validation set.
- Create Seed Ensemble (42-46).
- Perform Ablation study with 10 variants.
- Generate prediction reports only after obtaining actual metrics.
- Performance goals: Macro-F1 on locked test must increase by at least 0.01 or Recall Low must increase significantly without dropping Macro-F1 by more than 0.01, or xAPI improved clearly vs baseline 0.7850.
- If V27 is not better than baseline, keep baseline and report V27 as an extended experiment (no data fabrication).

## User Context
- **Last user request**: Launch the V27 model improvement project for student performance prediction (CNN-BiLSTM + Context MLP) across three datasets.
- **Pending clarifications**: none
- **Delivered results**:
  - Initialized V27 request recording in ORIGINAL_REQUEST.md.
  - Spawned the teamwork_preview_orchestrator (2d42b4cb-2222-43ba-9436-ae0707b291c0).
  - Setup Cron 1 (Progress Reporting) and Cron 2 (Liveness Check).
  - Spawned Victory Auditor (f6c96679-8f29-4ed3-9253-f582b436a4aa) after orchestrator claimed victory.

## Project Status
- **Phase**: auditing

## Victory Audit Status
- **Triggered**: yes
- **Verdict**: pending
- **Retry count**: 0

## Artifact Index
- c:\Huflit\kltn\.agents\ORIGINAL_REQUEST.md — Original user request
- c:\Huflit\kltn\.agents\teamwork_preview_orchestrator_v27 — Orchestrator V27 working directory
- c:\Huflit\kltn\.agents\victory_auditor_v27 — Victory Auditor working directory
