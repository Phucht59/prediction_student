# Context Document

## Project Background
The predecessor (generation 1) Project Orchestrator crashed due to resource exhaustion during the RA-HLPR implementation.
A worker subagent `worker_ra_hlpr_1` completed the implementation of the Risk-Aware Hybrid Learning Path Recommender (RA-HLPR) downstream system, ran unit tests (16/16 passed), and generated all expected files under `outputs/recommender/` on the `student-mat` dataset.
Our role is to verify the results, ensure they meet all requirements, and report back to the parent agent.

## Working Directory
- Project Root: `c:\Huflit\kltn`
- Agent Workspace: `c:\Huflit\kltn\.agents\teamwork_preview_orchestrator_ra_hlpr_1_gen2`

## Key Targets for Verification
1. `outputs/recommender/risk_predictions.csv`
2. `outputs/recommender/recommendation_results.csv`
3. `outputs/recommender/learning_paths.json`
4. `outputs/recommender/recommender_metrics.json`
5. `outputs/recommender/recommender_report.md`

## Key Requirements & Constraints
- Keep original resampling (ADASYN/SMOTE) and data preprocessing intact.
- Downstream predictor must not break the main CNN+BiLSTM pipeline or test metrics.
- Multi-label MLP functions as `RiskDiagnosisHead` predicting 6 risks.
- Hybrid scoring based on multi-criteria weights.
- 4-week structured learning paths generated for students.
- Logic rules for weak labeling must be transparently documented.
- No code modification allowed for the orchestrator.
