# Final Learning Path Recommender Report

## Goal
The recommender module generates risk-aware learning path recommendations from the finalized CNN-BiLSTM prediction outputs. It is not collaborative filtering, because the available datasets do not contain user-item interaction histories or post-recommendation feedback.

The module is downstream of prediction:

`CNN-BiLSTM probabilities -> risk diagnosis -> intervention ranking -> 4-week learning path`

## Pipeline Architecture
- **Prediction model**: finalized CNN-BiLSTM outputs `p_low`, `p_medium`, and `p_high`.
- **RiskDiagnosisHead**: maps observable academic/context features and class probabilities to multi-label risk scores.
- **CandidateGenerator**: filters interventions with prediction-aware thresholds. Low predictions use lower thresholds; High predictions use higher thresholds to avoid excessive remediation.
- **HybridScorer**: ranks candidates with adaptive weights based on predicted class, probability confidence, and maximum diagnosed risk.
- **PathPlanner**: schedules top interventions into a 4-week plan with risk band, plan intensity, top risks, and weekly actions.

## Fit With Thesis Scope
The thesis requires a recommendation component that uses prediction results to support individualized learning pathways. This implementation uses the CNN-BiLSTM probability vector as the main signal, combines it with operational risk diagnosis, then personalizes interventions by predicted class and risk scores. Machine-learning baselines are not used as teachers, distillation sources, pseudo-label sources, baseline probability sources, or feature-importance sources.

## Risk Definitions
Student datasets use six operational risk factors:

| Code | Risk factor | Operational signals |
|---|---|---|
| R1 | Low prior performance | failures, G1 |
| R2 | Declining trend | G2 lower than G1 |
| R3 | Attendance risk | absences |
| R4 | Low engagement | goout, freetime, activities |
| R5 | Insufficient study time | studytime |
| R6 | High failure probability | failures, G1/G2 level and trend; not G3 |

xAPI uses three observable risk factors:

| Code | Risk factor | Operational signals |
|---|---|---|
| R3 | Attendance risk | StudentAbsenceDays |
| R4 | Low engagement | VisITedResources, raisedhands, Discussion, AnnouncementsView |
| R6 | High failure probability | attendance, engagement, parent/school support; not true Class |

## Intervention Catalog
The intervention catalog covers attendance, study planning, LMS engagement, peer/group support, remedial practice, parent/school support, and enrichment for stable/high students.

## Scoring Formula
For each candidate intervention:

```text
score =
w1 * risk_match
+ w2 * performance_need
+ w3 * difficulty_fit
+ w4 * time_fit
+ w5 * prerequisite_fit
+ w6 * expected_effect
```

Weights are adaptive: Low/high-risk students prioritize `risk_match` and `performance_need`; Medium uses balanced weights; High/stable prioritizes enrichment, difficulty fit, prerequisites, and expected effect.

Each recommendation keeps a score breakdown and prediction context: `predicted_class`, `p_low`, `p_medium`, `p_high`, `max_diagnosed_risk`, and `adjusted_capacity_hours`.

## Learning Path
Each student receives a 4-week plan: Week 1 Stabilize, Week 2 Practice, Week 3 Reinforce, Week 4 Evaluate & Adjust.

## Evaluation Metrics
The recommender is evaluated offline against weak-supervision/rule-based reference labels. The full xAPI and Student-Por pipelines were refreshed in this run. Student-Mat is pending a refreshed full run because of missing/inconsistent final prediction feature metadata.

| Dataset | Risk Macro F1 | Risk Micro F1 | Precision@3 | Recall@3 | NDCG@3 | Coverage@3 | Risk Coverage | Workload Std | Difficulty Progression | Prereq Violation |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| xapi | 0.9831 | 0.9813 | 0.7014 | 0.4719 | 0.8407 | 0.6875 | 0.9306 | 1.3152 | 0.7396 | 0.0021 |
| student-por | 0.9359 | 0.9094 | 0.8462 | 0.3870 | 0.8800 | 1.0000 | 0.9335 | 1.4751 | 0.6410 | 0.0205 |

Student-Mat status: Current full run blocked because final Student-Mat prediction metadata is missing/inconsistent: models/saved/final/student-mat_3class_ensemble_features.json is absent and checkpoint input shape does not match regenerated feature selection. Existing outputs/recommender/student-mat files were not refreshed in this run.

## Case Studies
### xapi - High Risk (Struggling) Student (Test Index 1)
- Predicted class/probability: Class 0 (Low) - Probabilities: [Low: 0.59, Medium: 0.40, High: 0.00]
- Main risks: R3_ATTENDANCE_RISK=0.00, R4_LOW_ENGAGEMENT=1.00, R6_HIGH_FAILURE_PROBABILITY=1.00
- Top 3 interventions: Daily LMS Resource Checklist (0.97), Parent-Teacher Engagement Sync (0.97), Family Progress Contract (0.97)
- Path: Week 1: Stabilize, Week 2: Practice, Week 3: Reinforce, Week 4: Evaluate & Adjust
### xapi - Moderate Risk (Average) Student (Test Index 0)
- Predicted class/probability: Class 1 (Medium) - Probabilities: [Low: 0.00, Medium: 0.69, High: 0.30]
- Main risks: R3_ATTENDANCE_RISK=0.00, R4_LOW_ENGAGEMENT=0.00, R6_HIGH_FAILURE_PROBABILITY=0.00
- Top 3 interventions: Daily Attendance Monitoring (0.52), Daily LMS Resource Checklist (0.52), Academic Counselor Consultation (0.52)
- Path: Week 1: Stabilize, Week 2: Practice, Week 3: Reinforce, Week 4: Evaluate & Adjust
### xapi - Stable (High Performer) Student (Test Index 4)
- Predicted class/probability: Class 2 (High) - Probabilities: [Low: 0.00, Medium: 0.10, High: 0.90]
- Main risks: R3_ATTENDANCE_RISK=0.00, R4_LOW_ENGAGEMENT=0.00, R6_HIGH_FAILURE_PROBABILITY=0.00
- Top 3 interventions: Advanced Subject Seminar (0.92)
- Path: Week 1: Stabilize, Week 2: Practice, Week 3: Reinforce, Week 4: Evaluate & Adjust
### student-por - High Risk (Struggling) Student (Test Index 6)
- Predicted class/probability: Class 0 (Low) - Probabilities: [Low: 0.73, Medium: 0.27, High: 0.00]
- Main risks: R1_LOW_PRIOR_PERFORMANCE=1.00, R2_DECLINING_TREND=1.00, R3_ATTENDANCE_RISK=0.00
- Top 3 interventions: Parent-Teacher Engagement Sync (0.98), Family Progress Contract (0.98), Daily LMS Resource Checklist (0.97)
- Path: Week 1: Stabilize, Week 2: Practice, Week 3: Reinforce, Week 4: Evaluate & Adjust
### student-por - Moderate Risk (Average) Student (Test Index 2)
- Predicted class/probability: Class 1 (Medium) - Probabilities: [Low: 0.16, Medium: 0.81, High: 0.03]
- Main risks: R1_LOW_PRIOR_PERFORMANCE=0.00, R2_DECLINING_TREND=0.00, R3_ATTENDANCE_RISK=0.00
- Top 3 interventions: Guided Discussion Prompts (0.81), Facilitated Study Group (0.80), Daily LMS Resource Checklist (0.78)
- Path: Week 1: Stabilize, Week 2: Practice, Week 3: Reinforce, Week 4: Evaluate & Adjust
### student-por - Stable (High Performer) Student (Test Index 0)
- Predicted class/probability: Class 2 (High) - Probabilities: [Low: 0.00, Medium: 0.27, High: 0.73]
- Main risks: R1_LOW_PRIOR_PERFORMANCE=0.00, R2_DECLINING_TREND=0.00, R3_ATTENDANCE_RISK=0.00
- Top 3 interventions: Advanced Subject Seminar (0.61)
- Path: Week 1: Stabilize, Week 2: Practice, Week 3: Reinforce, Week 4: Evaluate & Adjust

## Technical Guardrails
- Final prediction champions are unchanged.
- No large prediction training was rerun.
- No ML baseline is used as teacher, distillation source, pseudo-label source, baseline probability source, or feature-importance source.
- No true locked-test label is used for operational recommendation generation.
- Student weak labels do not use G3 for risk assignment.
- xAPI weak labels do not use true Class for risk assignment.
- The recommender is not collaborative filtering.
