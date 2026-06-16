# Final Recommendation Model Design

## Goal

The recommendation module is a downstream component of the academic-performance prediction model. It does not replace the CNN-BiLSTM predictor. Its role is to convert the predicted class probabilities and observable student profile signals into a risk-aware 4-week learning path.

## Final structure

The updated recommendation pipeline follows this flow:

1. CNN-BiLSTM prediction outputs class probabilities: Low, Medium and High.
2. Observable profile and engagement signals generate auditable weak risk labels.
3. A small RiskDiagnosisHead learns risk probabilities from student features plus predicted class probabilities.
4. CandidateGenerator filters the intervention catalog using predicted class and diagnosed risk probabilities.
5. HybridScorer ranks interventions using adaptive weights.
6. PathPlanner builds a 4-week staged learning path.

## Academic risks

For Student datasets, the recommender considers six risks:

- R1_LOW_PRIOR_PERFORMANCE
- R2_DECLINING_TREND
- R3_ATTENDANCE_RISK
- R4_LOW_ENGAGEMENT
- R5_INSUFFICIENT_STUDY_TIME
- R6_HIGH_FAILURE_PROBABILITY

For xAPI, only observable behavior risks are used:

- R3_ATTENDANCE_RISK
- R4_LOW_ENGAGEMENT
- R6_HIGH_FAILURE_PROBABILITY

The xAPI risk rules do not use the true `Class` label. This keeps the recommendation module usable in real prediction settings and prevents the recommender from depending on final labels during operation.

## Prediction-aware filtering

The CandidateGenerator now uses different risk thresholds depending on the predicted performance class:

- Low prediction: lower threshold, prioritizing early intervention.
- Medium prediction: balanced threshold.
- High prediction: higher threshold, avoiding unnecessary remedial items unless risk is strong.

This makes recommendations more directly tied to prediction output instead of being a disconnected rule list.

## Prediction-aware scoring

HybridScorer now adapts scoring weights based on predicted class probabilities and maximum diagnosed risk:

- High-risk/Low prediction: emphasizes risk match and performance need.
- Medium prediction: balances remediation and reinforcement.
- Stable/High prediction: favors difficulty fit, prerequisites and enrichment.

The scorer still remains transparent because each recommendation stores the score breakdown: risk match, performance need, difficulty fit, time fit, prerequisite fit and expected effect.

## xAPI-specific improvements

The intervention catalog now includes xAPI-oriented support actions:

- Absence Recovery Pack
- Daily LMS Resource Checklist
- Guided Discussion Prompts
- Family Progress Contract

These interventions target attendance risk, low engagement and high failure probability derived from observable LMS behavior and support signals.

## Learning path

The PathPlanner returns:

- risk_band: High, Moderate or Stable
- plan_intensity: intensive, guided or maintenance
- top_risks
- max_risk_score
- 4 weekly stages: Stabilize, Practice, Reinforce, Evaluate & Adjust

The final path is designed for thesis reporting: it is interpretable, linked to the prediction output, and does not require a separate collaborative-filtering dataset.

## Notes for thesis writing

This is not a collaborative recommender system. It is a risk-aware learning-path recommendation module built on top of CNN-BiLSTM prediction results. That choice is appropriate because the available datasets do not contain real user-item interaction histories or feedback labels for recommendation learning.
