# Phase 8 EBM training

Target is `expected_relevance` from frozen Phase 7 silver labels. NO_WEAK_EVIDENCE is never trained as 0.
Panel B was not used for training, CV, or hyperparameter selection.

## Feature contract

Approved features: `stage_code, risk_probability, inactive_streak, active_days_ratio, recent_activity, activity_trend, assessment_completion, missing_assessments, quiz_activity, vle_available`.
`course_progress` is `FEATURE_EXCLUDED_REDUNDANT_STAGE` because it equals stage/100.
`risk_band` is excluded as a discretized duplicate of `risk_probability`.

## A4 feasibility audit

OLD_RULE: Progress Monitoring inherited Content Review UNKNOWN / `CONTENT_AVAILABILITY_UNOBSERVED`.
NEW_RULE: `recommendation.feasibility.v2` marks A4 FEASIBLE / `PROGRESS_STATE_OBSERVED`.
Historical v1 feasibility artifacts were not mutated.

## Training rows and selected EBM configs

| Action | Rows | Excluded no-evidence | max_bins | interactions | min_samples_leaf | CV folds |
|---|---:|---:|---:|---:|---:|---:|
| assessment_recovery | 141 | 359 | 16 | 0 | 10 | 5 |
| progress_monitoring | 500 | 0 | 32 | 0 | 5 | 5 |
| re_engagement | 500 | 0 | 32 | 3 | 10 | 5 |
| retrieval_practice | 311 | 189 | 16 | 3 | 5 | 5 |
| study_planning | 500 | 0 | 32 | 3 | 10 | 5 |

## OOF regression metrics (confidence-weighted EBM)

| Action | MAE | RMSE | Weighted MAE | Spearman | Status |
|---|---:|---:|---:|---:|---|
| assessment_recovery | 0.036543 | 0.099083 | 0.027726 | 0.612478 | `PASS` |
| progress_monitoring | 0.367609 | 0.454298 | 0.365932 | 0.813916 | `PASS_WITH_WARNING` |
| re_engagement | 0.256669 | 0.355348 | 0.239751 | 0.951003 | `PASS` |
| retrieval_practice | 0.123467 | 0.152801 | 0.125006 | 0.812272 | `REVIEW` |
| study_planning | 0.165778 | 0.232322 | 0.157556 | 0.891528 | `PASS` |

A1 has low n (141). A5 remains REVIEW. A4 preserves the Gemini-family weak-source warning.

## Weighted vs unweighted ablation

| Action | Weighted MAE | Unweighted MAE |
|---|---:|---:|
| assessment_recovery | 0.036543 | 0.042554 |
| progress_monitoring | 0.367609 | 0.366492 |
| re_engagement | 0.256669 | 0.261232 |
| retrieval_practice | 0.123467 | 0.124277 |
| study_planning | 0.165778 | 0.166899 |

Primary models remain confidence-weighted. Ablation uses the same selected config without sample weights.

## Panel A OOF ranking diagnostic

This is DEVELOPMENT diagnostic only, not a final test.

| Model | NDCG@3 | P@1 | Recall@3 | MRR | Pairwise |
|---|---:|---:|---:|---:|---:|
| EBM | 0.974858 | 0.426000 | 0.934576 | 0.958333 | 0.816867 |
| ACTION_STAGE_PRIOR | 0.842691 | 0.398000 | 0.825292 | 0.916667 | 0.577200 |
| RIDGE | 0.968632 | 0.416000 | 0.919225 | 0.946637 | 0.804600 |
| RANDOM_FOREST | 0.977397 | 0.432000 | 0.939693 | 0.967105 | 0.831533 |

## Global top terms

- `assessment_recovery`: assessment_completion=0.0128, quiz_activity=0.0056, missing_assessments=0.0049, risk_probability=0.0044, activity_trend=0.0043
- `progress_monitoring`: risk_probability=0.3309, missing_assessments=0.1626, assessment_completion=0.1357, activity_trend=0.0692, recent_activity=0.0619
- `re_engagement`: recent_activity=0.4055, risk_probability=0.3083, active_days_ratio=0.2421, inactive_streak=0.2348, activity_trend=0.1232
- `retrieval_practice`: quiz_activity=0.1864, risk_probability=0.0359, assessment_completion=0.0299, active_days_ratio=0.0200, activity_trend=0.0177
- `study_planning`: risk_probability=0.1488, recent_activity=0.1141, active_days_ratio=0.0948, activity_trend=0.0940, assessment_completion=0.0751
