# Scope: Downstream RA-HLPR System Recommender

## Architecture
- CNN-BiLSTM performance predictor: Outputs academic performance predictions (class probabilities, features) as downstream inputs.
- RiskDiagnosisHead (refactored MLP): Receives features & class probabilities, predicts R1-R6 multi-label risks.
- Intervention Knowledge Base: CSV files cataloging intervention items and mapping risks to them.
- Hybrid Scorer: Scores interventions using performance needs, risk match, difficulty, time constraints, prerequisites, and expected effect.
- Learning Path Planner (`path_planner.py`): Allocates recommended interventions into a 4-week learning path.
- Recommender Pipeline Script (`scripts/run_recommender_pipeline.py`): Orchestrates loading predictions, weak-labeling, training Risk Head, scoring, path planning, evaluation, and reporting.

## Milestones
| # | Name | Scope | Dependencies | Status |
|---|------|-------|-------------|--------|
| 1 | Exploration & Baseline Analysis | Understand data inputs, existing MLP model structure, existing data pipeline outputs, and predictions data structure | None | DONE |
| 2 | Risk Diagnosis & Weak Labeling | Implement weak-labeling rules, refactor MLP to RiskDiagnosisHead, train with BCEWithLogitsLoss + pos_weight, and write explanation file | M1 | DONE |
| 3 | Knowledge Base & Hybrid Scorer | Create intervention_catalog.csv, risk_intervention_mapping.csv, and build Hybrid Scorer | M2 | DONE |
| 4 | Path Planner & Recommender Pipeline | Build path_planner.py, implement scripts/run_recommender_pipeline.py with student-mat dataset support | M3 | DONE |
| 5 | Evaluation & Output Generation | Implement metrics calculations (Risk Diagnosis, Ranking, Path Quality) and generate target outputs in outputs/recommender/ | M4 | DONE |
| 6 | Final Review & Integrity Audit | Run full end-to-end tests, inspect generated reports, verify compliance with non-interference constraints, and run Forensic Auditor checks | M5 | DONE |

## Interface Contracts
- Input from CNN-BiLSTM: Prediction output CSV containing predicted_label, class_probabilities, confidence, and student features.
- RiskDiagnosisHead input: Student features concatenated with class probabilities. Output: Multi-label predictions for R1-R6 risks.
- LearningPathPlanner input: Top-K scored interventions for a student. Output: 4-week structured learning path JSON.
