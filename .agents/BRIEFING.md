# BRIEFING — 2026-06-15T10:11:00Z

## Mission
Orchestrate the refactoring and development of the downstream RA-HLPR (Risk-Aware Hybrid Learning Path Recommender) system.

## 🔒 My Identity
- Archetype: sentinel
- Working directory: c:\Huflit\kltn\.agents
- Orchestrator: da19f9da-92c3-4713-82c6-4444ea757405
- Victory Auditor: TBD

## 🔒 Key Constraints
- No technical decisions — relay only
- Victory Audit is MANDATORY before reporting completion
- Keep pre-processing and resampling logic strictly unchanged in src/data_pipeline.py and src/train_pipeline.py
- Do not modify or break the existing CNN+BiLSTM prediction pipeline or test metrics
- Recommend system must operate completely as a downstream module
- Refactor PyTorch MLP model only as Risk Diagnosis Head, not the main recommender
- Intervention catalog must contain realistic education actions
- No metric fabrication. Do not report evaluation table for datasets that haven't been run.
- Do not call it collaborative filtering if there's no user-item interaction data.
- Do not call it knowledge graph if no actual graph is constructed.
- Do not use risks without corresponding features in the dataset.

## User Context
- **Last user request**: Implement the RA-HLPR system refactoring as a downstream module, defining 6 specific risk rules, a 10+ item intervention catalog, a hybrid scorer, candidate generator, path planner, explanation module, evaluation metrics, and run-pipeline scripts, followed by report updates in generate_doc.py.
- **Pending clarifications**: none
- **Delivered results**:
  - Initial setup and delegation to the Project Orchestrator (da19f9da-92c3-4713-82c6-4444ea757405).

## Project Status
- **Phase**: in progress

## Victory Audit Status
- **Triggered**: no
- **Verdict**: pending
- **Retry count**: 0

## Artifact Index
- c:\Huflit\kltn\.agents\ORIGINAL_REQUEST.md — Original user request
