# A4 replacement evaluation (B1 vs B2)

Offline diagnostics only. This report does not choose a replacement and does not use mean relevance as a selection rule.
Candidate distinction: B1 monitors current progress/gaps; B2 seeks legitimate academic help. Neither is A3 Study Planning.

- Model configured: `gemini-3.5-flash-lite`
- Prompt version: `recommendation_a4_replacement_v1`
- Jobs: `50/50` completed; failed jobs: `0`
- Degeneracy diagnostic threshold: `0.95` of all labels in one class.
- State-variation diagnostic threshold: coverage range >= `0.10` or numeric-label mean range >= `0.25`; heuristic only.

## Overall candidate diagnostics

| Candidate | Numeric coverage | ABSTAIN rate | Distribution | Degeneracy | State variation features |
|---|---:|---:|---|---|---|
| B1_PROGRESS_MONITORING | 1.0000 | 0.0000 | `{'0': 54, '1': 178, '2': 212, '3': 56, 'ABSTAIN': 0}` | `NOT_FLAGGED` | `active_days_ratio, risk_probability, assessment_completion, inactive_streak, quiz_activity, missing_assessments, risk_band, recent_activity, activity_trend` |
| B2_ACADEMIC_HELP_SEEKING | 1.0000 | 0.0000 | `{'0': 211, '1': 143, '2': 99, '3': 47, 'ABSTAIN': 0}` | `NOT_FLAGGED` | `active_days_ratio, risk_probability, assessment_completion, inactive_streak, quiz_activity, missing_assessments, risk_band, recent_activity, activity_trend` |

## Coverage by stage

| Candidate | Group | N | Numeric coverage | ABSTAIN rate |
|---|---|---:|---:|---:|
| B1_PROGRESS_MONITORING | 20pct | 133 | 1.0000 | 0.0000 |
| B1_PROGRESS_MONITORING | 35pct | 129 | 1.0000 | 0.0000 |
| B1_PROGRESS_MONITORING | 50pct | 122 | 1.0000 | 0.0000 |
| B1_PROGRESS_MONITORING | 75pct | 116 | 1.0000 | 0.0000 |
| B2_ACADEMIC_HELP_SEEKING | 20pct | 133 | 1.0000 | 0.0000 |
| B2_ACADEMIC_HELP_SEEKING | 35pct | 129 | 1.0000 | 0.0000 |
| B2_ACADEMIC_HELP_SEEKING | 50pct | 122 | 1.0000 | 0.0000 |
| B2_ACADEMIC_HELP_SEEKING | 75pct | 116 | 1.0000 | 0.0000 |

## Coverage by risk_band

| Candidate | Group | N | Numeric coverage | ABSTAIN rate |
|---|---|---:|---:|---:|
| B1_PROGRESS_MONITORING | high | 109 | 1.0000 | 0.0000 |
| B1_PROGRESS_MONITORING | low | 272 | 1.0000 | 0.0000 |
| B1_PROGRESS_MONITORING | medium | 119 | 1.0000 | 0.0000 |
| B2_ACADEMIC_HELP_SEEKING | high | 109 | 1.0000 | 0.0000 |
| B2_ACADEMIC_HELP_SEEKING | low | 272 | 1.0000 | 0.0000 |
| B2_ACADEMIC_HELP_SEEKING | medium | 119 | 1.0000 | 0.0000 |

## Relationship with observable Student State

Coverage and numeric-label variation are reported by the supplied state features only. A `NONE_DETECTED` result means the diagnostic did not observe the configured variation threshold; it is not evidence that a candidate is universally irrelevant.

### B1_PROGRESS_MONITORING
- `active_days_ratio`: coverage_range=`0.0000`, numeric_mean_range=`1.0950`, variation=`FLAG`
- `risk_probability`: coverage_range=`0.0000`, numeric_mean_range=`1.4800`, variation=`FLAG`
- `assessment_completion`: coverage_range=`0.0000`, numeric_mean_range=`1.0573`, variation=`FLAG`
- `inactive_streak`: coverage_range=`0.0000`, numeric_mean_range=`0.9928`, variation=`FLAG`
- `quiz_activity`: coverage_range=`0.0000`, numeric_mean_range=`0.4811`, variation=`FLAG`
- `missing_assessments`: coverage_range=`0.0000`, numeric_mean_range=`1.1642`, variation=`FLAG`
- `course_progress`: coverage_range=`0.0000`, numeric_mean_range=`0.0611`, variation=`NOT_FLAGGED`
- `stage`: coverage_range=`0.0000`, numeric_mean_range=`0.0969`, variation=`NOT_FLAGGED`
- `risk_band`: coverage_range=`0.0000`, numeric_mean_range=`1.3724`, variation=`FLAG`
- `recent_activity`: coverage_range=`0.0000`, numeric_mean_range=`3.0000`, variation=`FLAG`
- `activity_trend`: coverage_range=`0.0000`, numeric_mean_range=`3.0000`, variation=`FLAG`
- `vle_available`: coverage_range=`0.0000`, numeric_mean_range=`0.0000`, variation=`NOT_FLAGGED`
### B2_ACADEMIC_HELP_SEEKING
- `active_days_ratio`: coverage_range=`0.0000`, numeric_mean_range=`1.5850`, variation=`FLAG`
- `risk_probability`: coverage_range=`0.0000`, numeric_mean_range=`2.1680`, variation=`FLAG`
- `assessment_completion`: coverage_range=`0.0000`, numeric_mean_range=`1.6546`, variation=`FLAG`
- `inactive_streak`: coverage_range=`0.0000`, numeric_mean_range=`1.4772`, variation=`FLAG`
- `quiz_activity`: coverage_range=`0.0000`, numeric_mean_range=`0.6750`, variation=`FLAG`
- `missing_assessments`: coverage_range=`0.0000`, numeric_mean_range=`1.7264`, variation=`FLAG`
- `course_progress`: coverage_range=`0.0000`, numeric_mean_range=`0.0824`, variation=`NOT_FLAGGED`
- `stage`: coverage_range=`0.0000`, numeric_mean_range=`0.1012`, variation=`NOT_FLAGGED`
- `risk_band`: coverage_range=`0.0000`, numeric_mean_range=`2.1206`, variation=`FLAG`
- `recent_activity`: coverage_range=`0.0000`, numeric_mean_range=`2.0000`, variation=`FLAG`
- `activity_trend`: coverage_range=`0.0000`, numeric_mean_range=`3.0000`, variation=`FLAG`
- `vle_available`: coverage_range=`0.0000`, numeric_mean_range=`0.0000`, variation=`NOT_FLAGGED`

## Decision boundary

No candidate is selected automatically. Final replacement selection requires reviewing these diagnostics together with semantic distinction from A1/A2/A3/A5 and supportability in the current Student State.
