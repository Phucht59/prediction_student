# Project Context - RA-HLPR Refactoring

## Environment
- OS: Windows
- Workspaces: `c:\Huflit\kltn` (CorpusName: `Phucht59/prediction_student`)
- App Data Directory: `C:\Users\THPhu\.gemini\antigravity`

## Project Specifications & Constraints
- Predictor model: CNN-BiLSTM + Context MLP must remain intact and untouched.
- RA-HLPR is a downstream module.
- MLP now acts as RiskDiagnosisHead.
- 6 Risks:
  - R1_LOW_PRIOR_PERFORMANCE
  - R2_DECLINING_TREND
  - R3_ATTENDANCE_RISK
  - R4_LOW_ENGAGEMENT
  - R5_INSUFFICIENT_STUDY_TIME
  - R6_HIGH_FAILURE_PROBABILITY
- Intervention catalog: at least 10 items in `data/recommender/intervention_catalog.csv` with specific headers.
- Hybrid Scorer weights:
  - risk_match: 0.3
  - performance_need: 0.2
  - difficulty_fit: 0.15
  - time_fit: 0.15
  - prerequisite_fit: 0.1
  - expected_effect: 0.1
- Path Planner: 4-week roadmap (Stabilize, Practice, Reinforce, Evaluate & Adjust).
- Weak label rules must use features from the dataset. No risks without features.
