# Final Learning Path Recommender Report

## Goal
The recommender module generates risk-aware learning path recommendations from the finalized CNN-BiLSTM prediction outputs. It is not collaborative filtering, because the available datasets do not contain user-item interaction histories or post-recommendation feedback.

The module is downstream of prediction:

`CNN-BiLSTM probabilities -> risk diagnosis -> intervention ranking -> 4-week learning path`

## Pipeline Architecture
- **Prediction model**: finalized CNN-BiLSTM outputs `p_low`, `p_medium`, and `p_high`.
- **RiskDiagnosisHead**: maps observable academic/context features and class probabilities to multi-label risk scores.
- **CandidateGenerator**: filters interventions with prediction-aware thresholds and `applicable_kind` (`student`, `xapi`, `both`). Low predictions use lower thresholds; High predictions use higher thresholds to avoid excessive remediation. When no risk is active and the prediction is Medium/High, only general/light reinforcement items are considered.
- **HybridScorer**: ranks candidates with adaptive weights based on predicted class, probability confidence, maximum diagnosed risk, dataset kind, and intervention type.
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
The intervention catalog is dataset-aware through `applicable_kind`:

| Group | Example interventions | Scope |
|---|---|---|
| attendance | Daily Attendance Monitoring, Absence Recovery Pack | both/xAPI when R3 is active |
| study planning | Time Management Workshop, Standard Practice Plan | both |
| LMS engagement | LMS Resource Checklist, Maintain LMS Engagement, Interactive Quizzing | xAPI |
| peer/group support | Peer-Led Study Tutoring, Facilitated Study Group | student/both |
| remedial practice | Targeted Practice Exercises, Remedial Topic Bootcamps, Academic Coaching | student |
| parent/school support | Parent-Teacher Sync, Family Progress Contract | both, scored high only when R6/support risk is active |
| enrichment/light maintenance | Advanced Seminar, Weekly Progress Review, Optional Discussion Prompt | both/xAPI |

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
+ rule_adjustment
```

Weights are adaptive: Low/high-risk students prioritize `risk_match` and `performance_need`; Medium uses balanced weights; High/stable prioritizes enrichment, difficulty fit, prerequisites, and expected effect. Rule adjustments enforce domain logic: Student R1/R2 boosts academic remediation, xAPI R4 boosts LMS/resource/discussion actions, and parent/family support is penalized unless R6/support risk is active.

Each recommendation keeps a score breakdown and prediction context: `predicted_class`, `p_low`, `p_medium`, `p_high`, `max_diagnosed_risk`, and `adjusted_capacity_hours`.

## Learning Path
Each student receives a 4-week plan: Week 1 Stabilize, Week 2 Practice, Week 3 Reinforce, Week 4 Evaluate & Adjust.

## Evaluation Metrics
The recommender is evaluated offline against weak-supervision/rule-based reference labels. The full xAPI and Student-Por pipelines were refreshed in this run. Student-Mat is pending a refreshed full run because the final prediction checkpoint metadata is missing/inconsistent.

| Dataset | Risk Macro F1 | Risk Micro F1 | Precision@3 | Recall@3 | NDCG@3 | Coverage@3 | Risk Coverage | Workload Std | Difficulty Progression | Prereq Violation |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| xapi | 0.9831 | 0.9813 | 0.6840 | 0.4720 | 0.8229 | 0.6500 | 0.8958 | 1.1210 | 0.7153 | 0.0000 |
| student-por | 0.9359 | 0.9094 | 0.6641 | 0.3185 | 0.7455 | 0.5500 | 0.9508 | 1.3137 | 0.6000 | 0.0449 |

Student-Mat status: pending full run because missing final prediction checkpoint metadata: models/saved/final/student-mat_3class_ensemble_features.json. The available Student-Mat checkpoint input shape does not match regenerated feature selection, so outputs/recommender/student-mat was not refreshed in this run.

## Case Studies
### xapi - High Risk (Struggling) Student (Test Index 1)
- Predicted class/probability: Class 0 (Low) - Probabilities: [Low: 0.59, Medium: 0.40, High: 0.00]
- Main risks: R4_LOW_ENGAGEMENT=1.00, R6_HIGH_FAILURE_PROBABILITY=1.00, R3_ATTENDANCE_RISK=0.00
- Top 3 interventions: Daily LMS Resource Checklist (1.00), Guided Discussion Prompts (1.00), LMS Interactive Quizzing (1.00)
- Path: Week 1: Stabilize, Week 2: Practice, Week 3: Reinforce, Week 4: Evaluate & Adjust
### xapi - Moderate Risk (Average) Student (Test Index 0)
- Predicted class/probability: Class 1 (Medium) - Probabilities: [Low: 0.00, Medium: 0.69, High: 0.30]
- Main risks: R3_ATTENDANCE_RISK=0.00, R4_LOW_ENGAGEMENT=0.00, R6_HIGH_FAILURE_PROBABILITY=0.00
- Top 3 interventions: Standard Practice Plan (0.80), Weekly Progress Review (0.80), Maintain LMS Engagement (0.78)
- Path: Week 1: Stabilize, Week 2: Practice, Week 3: Reinforce, Week 4: Evaluate & Adjust
### xapi - Stable (High Performer) Student (Test Index 4)
- Predicted class/probability: Class 2 (High) - Probabilities: [Low: 0.00, Medium: 0.10, High: 0.90]
- Main risks: R4_LOW_ENGAGEMENT=0.00, R3_ATTENDANCE_RISK=0.00, R6_HIGH_FAILURE_PROBABILITY=0.00
- Top 3 interventions: Advanced Subject Seminar (1.00), Maintain LMS Engagement (0.85), Standard Practice Plan (0.85)
- Path: Week 1: Stabilize, Week 2: Practice, Week 3: Reinforce, Week 4: Evaluate & Adjust
### student-por - High Risk (Struggling) Student (Test Index 6)
- Predicted class/probability: Class 0 (Low) - Probabilities: [Low: 0.73, Medium: 0.27, High: 0.00]
- Main risks: R1_LOW_PRIOR_PERFORMANCE=1.00, R2_DECLINING_TREND=1.00, R6_HIGH_FAILURE_PROBABILITY=1.00
- Top 3 interventions: Peer-Led Study Tutoring (1.00), Targeted Practice Exercises (1.00), Biweekly Academic Coaching (1.00)
- Path: Week 1: Stabilize, Week 2: Practice, Week 3: Reinforce, Week 4: Evaluate & Adjust
### student-por - Moderate Risk (Average) Student (Test Index 2)
- Predicted class/probability: Class 1 (Medium) - Probabilities: [Low: 0.16, Medium: 0.81, High: 0.03]
- Main risks: R4_LOW_ENGAGEMENT=0.65, R2_DECLINING_TREND=0.00, R6_HIGH_FAILURE_PROBABILITY=0.00
- Top 3 interventions: Facilitated Study Group (0.80), Daily Attendance Monitoring (0.53), Academic Counselor Consultation (0.53)
- Path: Week 1: Stabilize, Week 2: Practice, Week 3: Reinforce, Week 4: Evaluate & Adjust
### student-por - Stable (High Performer) Student (Test Index 0)
- Predicted class/probability: Class 2 (High) - Probabilities: [Low: 0.00, Medium: 0.27, High: 0.73]
- Main risks: R4_LOW_ENGAGEMENT=0.49, R3_ATTENDANCE_RISK=0.00, R1_LOW_PRIOR_PERFORMANCE=0.00
- Top 3 interventions: Advanced Subject Seminar (0.67), Standard Practice Plan (0.54), Weekly Progress Review (0.54)
- Path: Week 1: Stabilize, Week 2: Practice, Week 3: Reinforce, Week 4: Evaluate & Adjust

## Sanity Checks After Dataset-Aware Filtering
- Student-Por high-risk R1/R2 case now ranks academic remediation in the top 3.
- xAPI no-risk Medium case now ranks light/general items instead of attendance/counselor interventions.
- xAPI high-risk low-engagement case ranks LMS/resource/discussion interventions before family support.

## Technical Guardrails
- Final prediction champions are unchanged.
- No large prediction training was rerun.
- No ML baseline is used as teacher, distillation source, pseudo-label source, baseline probability source, or feature-importance source.
- No true locked-test label is used for operational recommendation generation.
- Student weak labels do not use G3 for risk assignment.
- xAPI weak labels do not use true Class for risk assignment.
- The recommender is not collaborative filtering.
