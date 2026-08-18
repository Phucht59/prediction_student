# Phase 1-2 Feature Mapping

Source of truth: current `hybrid_recomend` prediction and Phase7 feature artifacts.

| Semantic feature | Actual source feature(s) | Dataset | Transformation | Temporal validity | Missing policy | Status |
|---|---|---|---|---|---|---|
| risk_probability | prediction artifact score, Hybrid mean over seeds | OULAD | mean frozen Hybrid score | prediction artifact; target unused | not imputed | PASS |
| risk_band | risk_probability + config 0.33/0.66 | OULAD | operational low/medium/high mapping | downstream only; not prediction threshold | configurable | PASS |
| uncertainty | NONE | OULAD | not derived | UNAVAILABLE | omitted | UNAVAILABLE |
| inactive_streak | current_inactivity_streak | OULAD | direct aggregate channel | strict pre-cutoff view | valid zero | PASS |
| active_days_ratio | active_days + week_exposure_fraction | OULAD | sum active days / observed days | strict pre-cutoff view | valid zero | PASS |
| recent_activity | recent_activity | OULAD | direct aggregate channel | strict pre-cutoff view | valid zero | PASS |
| activity_trend | activity_trend | OULAD | direct aggregate channel | strict pre-cutoff view | valid zero | PASS |
| study_regularness | NONE | OULAD | not derived | UNAVAILABLE | omitted | UNAVAILABLE |
| assessment_completion | completion_rate | OULAD | direct aggregate channel | due/submission dates < cutoff | valid zero | PASS |
| missing_assessments | missed_due_count | OULAD | direct aggregate channel | due dates < cutoff | valid zero | PASS |
| upcoming_assessments | NONE | OULAD | not derived | future schedule not used | omitted | UNAVAILABLE |
| course_progress | view.progress | OULAD | cutoff fraction | stage contract | not imputed | PASS |
| content_coverage | NONE | OULAD | content clicks are not coverage | UNAVAILABLE | omitted | UNAVAILABLE |
| quiz_activity | quiz_activity temporal channel | OULAD | sum observed weeks | strict pre-cutoff view | valid zero | PASS |
| vle_available | temporal_mask | OULAD | observed window exists | registration-aware mask | boolean | PASS |
| content_available | NONE | OULAD | availability channel absent | UNAVAILABLE | omitted | UNAVAILABLE |
| quiz_available | NONE | OULAD | availability channel absent | UNAVAILABLE | omitted | UNAVAILABLE |

## Stage counts

| Stage | State rows | Eligible feature rows |
|---|---:|---:|
| 20pct | 26697 | 26697 |
| 35pct | 25606 | 25606 |
| 50pct | 24599 | 24599 |
| 75pct | 23159 | 23159 |

## UCI compatibility audit

| Semantic feature | UCI source | Status |
|---|---|---|
| risk_probability | Frozen Hybrid prediction artifact S0/S1/S2 | AVAILABLE FOR AUDIT ONLY |
| identity/stage/fold | record_id/group_id/outer_fold/module/presentation | AVAILABLE FOR AUDIT ONLY |
| engagement/VLE/content/quiz | NONE in UCI pipeline | UNAVAILABLE |
| assessment completion/missing/upcoming | NONE; grades are not completion | UNAVAILABLE |
| course progress | stage indicator only, not measured learning progress | UNAVAILABLE |
| uncertainty | NONE persisted | UNAVAILABLE |

## Provenance
- Prediction artifact SHA-256: `bd3ea11558fce882ae4371af1d6421b2a86f6adaf2ff57254c14b8ec54fca768`
- State artifact SHA-256: `60d9f7d2e2bb4307271ebaf329e6062d1ad4c8058863625f863edff50f55b162`
- `date_unregistration` is used only by the current Prediction eligibility contract, never copied as a state feature.
- `target`, `final_result`, and assessment `score` are not copied into the state artifact.
- OULAD enrollment_identity equals the existing record_id: sha256('oulad|code_module|code_presentation|id_student')[:24].
