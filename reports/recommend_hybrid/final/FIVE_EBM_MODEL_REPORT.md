# Five-EBM Explainable Action Ranker Report

## Architecture Summary
- **Model Type**: Five Independent Explainable Boosting Regressors (`FiveEBMRanker`)
- **Trained Actions**: `ASSESSMENT_COMPLETION, RECOVER_ENGAGEMENT, STUDY_REGULARITY, TARGETED_CONTENT_REVIEW, QUIZ_RETRIEVAL_PRACTICE`
- **Features Used**: `14` learner state features
- **Action ID Feature**: **Excluded** (Zero Action-Stage Identity Shortcut)
- **Score Calibration**: `IsotonicRegression` mapping predictions to shared relevance scale [0, 3]

## Action Models
1. `EBM_ASSESSMENT_COMPLETION`: Predicts assessment urgency relevance.
2. `EBM_RECOVER_ENGAGEMENT`: Predicts engagement recovery relevance.
3. `EBM_STUDY_REGULARITY`: Predicts study pattern regularity relevance.
4. `EBM_TARGETED_CONTENT_REVIEW`: Predicts topic review relevance.
5. `EBM_QUIZ_RETRIEVAL_PRACTICE`: Predicts quiz practice relevance.
