# Recommendation V2 — Scientific problem definition and optimization plan

## 1. Why the previous description was confusing

The repository previously mixed three different questions:

1. Does the learner need support now?
2. If support is needed, which action should be ranked first?
3. Would changing the related behaviour make the frozen Hybrid model predict less risk?

These questions have different populations, targets and metrics. A high ranking metric does not prove that the system knows when to issue a recommendation, and a lower simulated risk does not prove a causal improvement in grades.

Recommendation V2 therefore treats them as three explicit decision layers.

## 2. Decision layer A — Should the system intervene?

### Population

Every OULAD learner-course record that is operationally eligible at the 20%, 35%, 50% or 75% cutoff is evaluated. The population is not restricted to silver-positive groups.

### Runtime inputs

- Frozen Hybrid risk probability available at the cutoff.
- Predictive entropy.
- Seed disagreement when available.
- Current behaviour deficits derived only from information available at the cutoff.

### Offline target

The final at-risk outcome is used only for held-out offline evaluation. It is not available at runtime and never enters model features.

### Outputs

- `NO_ACTION`: current evidence does not justify an intervention.
- `BEHAVIOURAL_ACTION`: issue one ranked behavioural recommendation.
- `DEFER_TO_HUMAN`: evidence is high-risk but too uncertain or safety-sensitive for automation.

### Metrics

AUROC, PR-AUC, Brier, ECE, precision, recall, F1, balanced accuracy, specificity, confusion matrix, intervention rate, false-issue rate, missed-support rate, defer rate and selective risk-coverage are reported for the full population.

Policy thresholds are selected on validation only and frozen before test evaluation.

## 3. Decision layer B — Which action should be ranked first?

### Learned behavioural taxonomy

The five existing action slots are retained because they correspond to directly observable and modifiable OULAD behaviour families:

1. `ASSESSMENT_COMPLETION`
2. `STUDY_REGULARITY`
3. `VLE_ENGAGEMENT`
4. `QUIZ_OR_RETRIEVAL_PRACTICE`
5. `CONTENT_REVIEW`

These five slots are not the whole recommendation system. They are the subset suitable for a learned behavioural ranker.

### Governance routes outside the ranker

- `PROGRESS_MONITORING`: no immediate behavioural intervention; review at the next landmark.
- `DIAGNOSTIC_CHECK`: evidence is insufficient to choose a learning action.
- `INSTRUCTOR_CONTACT`: human academic support is required.
- `ADVISOR_ESCALATION`: critical or safety-sensitive cases require human review.

Human-support routes are policy decisions, not behavioural alternatives to be optimized by the same action head.

### Research candidate

`ASSESSMENT_TIMELINESS` is audited separately because submitting an assessment and submitting it on time are different behaviours. It must not be added as a sixth learned slot until prevalence, stage support, non-redundancy and expert interpretability pass.

### Ranking utility

The new ranker combines:

- calibrated action probability;
- observed need severity;
- frozen-Hybrid simulated risk reduction where available;
- evidence confidence;
- workload penalty;
- predictive uncertainty penalty.

Weights are selected on validation only. Test rows cannot tune weights or thresholds.

### Ranking metrics

Precision@1, Recall@1, Recall@3, NDCG@3, MRR, pairwise accuracy, action diversity, top-action concentration and positive-group coverage are reported. Results are compared with random, popularity, lowest-workload, severity-only, action-probability-only and risk-reduction-only baselines.

## 4. Decision layer C — Frozen-Hybrid intervention sensitivity

This is a model-based sensitivity analysis, not a causal intervention study.

### Procedure

For each action and each bounded strength:

1. Copy the learner's pre-cutoff raw 16-channel weekly behaviour sequence.
2. Modify only observed weeks and only channels directly related to the action.
3. Recompute all 47 dynamic channels using the canonical OULAD transformation.
4. Recompute the 161 temporal aggregate features.
5. Keep the four stage-context features and static variables unchanged.
6. Apply the original train-fitted checkpoint preprocessor.
7. Run the original frozen 5-seed Hybrid ensemble.
8. Compare simulated risk with baseline risk.

The model, checkpoint, cutoff, static inputs, outcome and future padding are immutable.

### Examples of bounded edits

- VLE engagement: increase recent clicks, active days and site coverage.
- Study regularity: convert recent inactive weeks into small purposeful sessions and recompute inactivity streaks.
- Retrieval practice: increase quiz/retrieval activity.
- Content review: increase content-oriented activity.
- Assessment completion: add one valid submission opportunity and related assessment activity without inventing a score.

### Metrics

Mean and median risk delta, fraction with lower risk, threshold-crossing fraction, monotonicity across intervention strengths, per-action/per-stage support, fold/seed stability and constraint violations are reported.

A negative risk delta means only that the frozen model responds favourably to the bounded input change. It does not prove that displaying the recommendation would produce that behaviour or improve the final grade.

## 5. Independent evidence and circularity controls

The action head is trained against scientific silver labels that partly depend on the same observed behaviours. Therefore perfect silver-label ranking is reported as consistency, not real-world accuracy.

Recommendation V2 adds:

- full-population intervention evaluation against held-out final risk outcome;
- feature-family ablations;
- ranking baselines;
- taxonomy coverage and redundancy audit;
- simulation evidence that is computed through the frozen Hybrid prediction path;
- exportable expert-review cases.

Expert review or prospective deployment remains necessary before production authorization.

## 6. Release interpretation

A valid V2 release may conclude:

> The system can be evaluated offline on when to support, which observable behavioural action to prioritize, and how the frozen risk model responds to bounded behaviour changes.

It may not conclude:

> Showing a recommendation is proven to improve a learner's grade.
