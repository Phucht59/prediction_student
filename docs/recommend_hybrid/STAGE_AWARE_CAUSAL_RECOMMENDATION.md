# Stage-Aware Causal Recommendation Extension

## 1. Objective

The existing conditional ranker answers two offline questions:

1. Which learner-state groups are eligible for support?
2. Which canonical action should be ranked highest for an eligible learner?

This extension adds a third observational question:

> For learners at the same prediction landmark, is the observed post-landmark behaviour consistent with an action associated with a better final outcome than comparable learners who did not show that behaviour?

The extension does not claim that displaying a recommendation caused the change. The system has not been deployed, so actual receipt, compliance, and user response are unavailable.

## 2. Four prediction landmarks

The neural ranker and the causal protocol use the same stage identity.

| Stage | Baseline information | Treatment-observation window | Outcome |
|---|---:|---:|---|
| `EARLY_20` | start through 20% | after 20% through 35% | final Pass/Fail |
| `EARLY_35` | start through 35% | after 35% through 50% | final Pass/Fail |
| `MIDDLE_50` | start through 50% | after 50% through 75% | final Pass/Fail |
| `LATE_75` | start through 75% | after 75% through course end | final Pass/Fail |

A causal trial is created at the stage that generated the recommendation. `MIDDLE_50` is not a universal default.

## 3. Recommendation lifecycle

The business rule is `LATEST_VALID_RECOMMENDATION_WINS`.

- A recommendation at a later stage updates or replaces the earlier action.
- Only one primary recommendation may be active for a learner at one time.
- Earlier actions remain in the audit history as `SUPERSEDED`.
- If the later stage abstains, the previous active action is closed.

This prevents recommendations from 20%, 35%, 50%, and 75% being interpreted as simultaneously active.

## 4. Treatment meaning

The treatment is not a UI event. It is observed behaviour after the landmark that is consistent with the canonical action.

| Action | Normalized observed measure | Minimum improvement |
|---|---|---:|
| `ASSESSMENT_COMPLETION` | `assessment_completion_rate` | 0.15 |
| `STUDY_REGULARITY` | `study_regularity_score` | 0.20 |
| `VLE_ENGAGEMENT` | `vle_active_day_rate` | 0.15 |
| `QUIZ_OR_RETRIEVAL_PRACTICE` | `retrieval_practice_rate` | 0.15 |
| `CONTENT_REVIEW` | `content_review_coverage` | 0.15 |

The minimum follow-up level is fitted on the train partition only. Validation and test rows replay the frozen treatment rule.

If an action-stage pair lacks measurable treatment, adequate treated/control counts, or overlap, its status is `CAUSAL_EVIDENCE_NOT_IDENTIFIABLE`.

## 5. Confounder control

Baseline features must be available no later than the prediction cutoff. The intended confounder set includes:

- prior assessment score and completion;
- on-time submission history;
- VLE clicks and active days;
- activity regularity and inactivity gaps;
- quiz/retrieval activity;
- course/module/presentation context;
- previous attempts and studied credits;
- frozen Hybrid risk probability;
- frozen Hybrid learner-state embedding;
- other pre-cutoff context already authorized by the prediction protocol.

Post-cutoff behaviour, the final outcome, and future-stage information are forbidden baseline features.

## 6. Estimator

Each action-stage trial uses:

1. grouped cross-fitting by learner;
2. a propensity model for observed treatment;
3. separate outcome models under treatment and control;
4. the AIPW doubly robust score;
5. a DR pseudo-outcome model for out-of-fold CATE;
6. propensity overlap trimming;
7. stabilized inverse-probability weights;
8. student-cluster percentile bootstrap.

The effect outputs are:

- ATE for the retained overlap population;
- CATE for each retained/evaluated row;
- predicted final outcome under control and treatment;
- propensity score;
- cross-fit fold;
- overlap-retention status.

## 7. Identifiability gates

The preregistered release gates are:

- at least 30 treated rows after trimming;
- at least 30 control rows after trimming;
- propensity within `[0.10, 0.90]`;
- no more than 30% of rows trimmed;
- weighted effective sample size at least 25% of retained rows;
- at least 90% of baseline features with absolute SMD below 0.10;
- no baseline feature with absolute SMD above 0.20;
- 1,000 student-cluster bootstrap iterations.

Failing any mandatory gate prohibits a causal-effect claim for that action-stage pair.

## 8. Causal issuance gate

The ranker remains the first authority. Causal evidence does not introduce a new action identity.

An action can be marked causally supported only when:

1. the conditional ranker authorizes and ranks the action;
2. the action-stage trial is identifiable;
3. the learner propensity lies in the overlap region;
4. individual estimated benefit is at least 0.05;
5. the stage-action bootstrap lower bound is not below zero.

Otherwise the service abstains with an explicit reason.

## 9. Hybrid imbalance sensitivity

The official Hybrid checkpoint remains frozen. Four identical linear prediction heads are compared on its frozen embeddings:

- `none`;
- `class_weight`;
- `smote`;
- `adasyn`.

SMOTE and ADASYN operate only on training embeddings. Validation selects the threshold. Test supplies final metrics. The study produces reporting evidence and cannot automatically replace the canonical checkpoint.

Metrics include ROC-AUC, PR-AUC, precision, recall, F1, balanced accuracy, specificity, Brier score, and the confusion matrix.

## 10. Leakage controls

The release must fail when any of these rules is violated:

- baseline information extends past the stage cutoff;
- the treatment window starts at or before the cutoff;
- the treatment window extends beyond the next landmark;
- final outcome appears in baseline features;
- one learner is split across cross-fit groups;
- treatment thresholds are fitted outside train;
- validation/test rows are synthetically resampled;
- test data selects the classification threshold;
- three-stage thresholds are used to authorize `LATE_75`.

## 11. Artifact contract

Local execution must create:

- `artifacts/recommend_hybrid/causal/input/frozen_embeddings.npz`;
- `artifacts/recommend_hybrid/causal/input/target_trials.npz`;
- `artifacts/recommend_hybrid/causal/imbalance/metrics.json`;
- `artifacts/recommend_hybrid/causal/target_trials/stage_action_effects.json`;
- `artifacts/recommend_hybrid/causal/target_trials/individual_effects.csv`;
- `artifacts/recommend_hybrid/causal/target_trials/manifest.json`;
- `reports/recommend_hybrid/causal/STAGE_AWARE_CAUSAL_VALIDATION.json`;
- `reports/recommend_hybrid/causal/STAGE_AWARE_CAUSAL_RESULTS.md`.

The target-trial archive requires aligned arrays:

- `features`;
- `treatment`;
- `outcome`;
- `groups`;
- `student_ids`;
- `stages`;
- `action_ids`;
- `baseline_progress`;
- `treatment_start_progress`;
- `treatment_end_progress`.

## 12. Allowed conclusion

When the validation gate passes, the supported conclusion is:

> Within the observed OULAD overlap population and under the stated causal assumptions, post-landmark behaviour consistent with selected actions has an estimated association/effect on final Pass probability using a cross-fitted doubly robust target-trial analysis.

The unsupported conclusion remains:

> Showing the recommendation to a learner is proven to improve the learner's grade.

That stronger conclusion requires deployment data, verified recommendation exposure/compliance, and preferably a randomized or otherwise defensible prospective intervention study.
