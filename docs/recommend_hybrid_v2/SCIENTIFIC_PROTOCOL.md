# Hybrid Risk-Guided Explainable Recommendation V2

## Status

`PROTOCOL_LOCKED_BEFORE_LOCAL_TRAINING`

This branch replaces only the recommendation decision layer. The frozen Hybrid CNN–BiLSTM architecture, checkpoints, grouped student splits, feature cutoffs, and prediction authority from commit `f2a06a7b77b2bf7519cc2ff70b3997c703ae6819` remain unchanged.

## Research question

For a learner classified as high risk by the frozen Hybrid model at one of four validated stages, which feasible learning action is most relevant to the learner's observed pre-cutoff state?

The recommendation module must not relearn whether the learner is at risk. Hybrid is the risk authority. The new module only performs action feasibility, relevance ranking, safe abstention, and learning-plan construction.

## Runtime pipeline

1. Build leakage-safe features using data strictly before the stage cutoff.
2. Run the frozen Hybrid ensemble and obtain risk probability, uncertainty, and seed disagreement.
3. Apply validation-selected risk stratification:
   - `LOW` -> `NO_ACTION`
   - `BORDERLINE` -> `MONITOR`
   - `HIGH` -> recommendation processing
4. Apply deterministic feasibility and contraindication filters.
5. Score each eligible canonical action with one independently trained explainable relevance model.
6. Calibrate scores on validation data and rank actions.
7. Apply a fail-closed safety and ambiguity router.
8. Emit `RECOMMEND`, `HUMAN_REVIEW`, `MONITOR`, or `NO_ACTION`.
9. Build a concrete learning plan with rationale, duration, measurable target, and review point.
10. Optionally run a constrained Hybrid plausibility simulator. Simulation is never causal evidence.

## Canonical action families

Exactly five automatic action families are allowed in V2:

1. `ASSESSMENT_COMPLETION`
2. `RECOVER_ENGAGEMENT`
3. `STUDY_REGULARITY`
4. `TARGETED_CONTENT_REVIEW`
5. `QUIZ_RETRIEVAL_PRACTICE`

`HUMAN_REVIEW`, `MONITOR`, and `NO_ACTION` are routes, not learning actions.

## Label protocol

Training targets are probabilistic scientific silver relevance labels, not expert ground truth.

Allowed label sources:

- literature-grounded labeling functions with explicit citation metadata;
- OULAD behavioral labeling functions using only pre-cutoff evidence;
- availability and contraindication rules;
- blinded LLM weak annotators that may abstain.

Disallowed label sources:

- final outcome as an action-ranking label;
- current action-head predictions;
- test-set statistics;
- post-cutoff behavior;
- causal ATE/CATE estimates that failed identifiability.

Each action receives an ordinal relevance target in `{0, 1, 2, 3}` or `ABSTAIN`. Snorkel or an equivalent train-only label model aggregates weak sources. Source confidence and conflict may weight training but must never be model features.

## Model family

Primary candidate: five separate Explainable Boosting relevance models, one per action. Raw `action_id` is not a feature of these models, preventing the previous action-prior shortcut.

Required comparators:

- rule severity ranking;
- global action popularity;
- action-stage-only model;
- logistic or linear relevance model;
- LambdaMART challenger;
- frozen four-stage neural action-head baseline.

The final model is the simplest model inside the statistically indistinguishable best-performing set. Neural complexity is not preferred by default.

## Split and tuning authority

- Split unit: student.
- Outer evaluation: three grouped folds.
- Repeated seeds: `42, 1201, 2026, 3407, 7319` where applicable.
- All preprocessing, label aggregation, thresholds, calibration, hyperparameters, feature selection, and simulator dose definitions are fitted or selected without test access.
- The final test is opened once after protocol lock.

## Primary metrics

Ranking primary endpoint: `NDCG@3`.

Secondary ranking endpoints:

- Precision@1;
- MRR;
- Recall@3;
- pairwise accuracy;
- invalid-action rate;
- action diversity;
- top-1 stability across folds and seeds;
- recommendation coverage and abstention.

Risk-policy reporting:

- PR-AUC and calibration of frozen Hybrid;
- low, borderline, and high-risk coverage;
- alerts per 1,000 learners at validation-selected budgets.

Simulator reporting:

- empirical-support rate;
- positive-response rate;
- median model-implied risk delta;
- adverse-response rate;
- seed consistency;
- dose consistency;
- placebo response.

## Mandatory scientific gates

A final recommendation model is ineligible unless all gates pass:

1. Student overlap across train/test is zero.
2. Post-cutoff feature violations are zero.
3. Invalid automatic action rate is zero.
4. Full-context ranking beats rule, popularity, and action-stage-only baselines.
5. Bootstrap confidence interval for the full-minus-action-stage-only NDCG difference excludes zero.
6. Context permutation causes a material ranking degradation.
7. Leave-one-label-source-out analysis does not reveal complete dependence on one source.
8. Results are reported per stage and module/presentation subgroup.
9. Score calibration and abstention policy are selected on validation only.
10. No expert, deployment, or causal effectiveness claim is made without corresponding evidence.

## Claim boundary

Permitted claim:

> The system provides an explainable, stage-aware offline ranking of feasible learning actions for learners identified as high risk by the frozen Hybrid model, evaluated against probabilistic weak-supervision targets and a held-out pseudo-expert benchmark.

Forbidden claims:

- recommendations are expert-validated ground truth;
- an offline metric of 1.0 is real-world accuracy;
- simulated risk reduction is a causal treatment effect;
- recommendations improve grades in deployment;
- LLM annotators are human experts.

## Release rule

Until prospective exposure, adherence, and outcome data exist, every V2 registry record must retain:

`runtime_authorized: false`
