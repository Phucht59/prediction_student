# Phase 4 constrained learning-plan pipeline

## Scope and authority

Phase 4 converts the immutable output of the Phase 3 evidence policy into a replayable learning plan. It does not change the frozen CNN-BiLSTM, checkpoint bytes, prediction outputs, routing, action priority, or expert-evaluation status. Phase 3 policy decisions are the only action source; no ranker, action score, pseudo-label, or effectiveness model is present.

## Runtime flow

`typed prediction context → cutoff-safe observed state → Phase 3 policy → deterministic selector → shared constraint solver → dataset plan builder → explanation/lineage → service → append-safe persistence`

`HybridRecommendationService` generates new plans and supports retrieve/replay. `generate_plan.py` accepts a JSON fixture, routes `student_mat`, `student_por`, or `oulad`, and supports `--dry-run`; a persistence directory is mandatory for non-dry runs.

## Contracts

`SelectedAction` records the Phase 3 priority, scheduled period, workload, reason codes, direct evidence, success criterion, human-contact requirement, and policy version. `LearningPlan` records dataset/student/course identity, requested cutoff, past prediction anchor, one of `FULL`, `PARTIAL`, `ABSTAIN`, or `EVALUATION_ONLY`, actions, workload, periods, explanation, complete authority lineage, versions, and creation time. ABSTAIN and EVALUATION_ONLY contracts reject non-empty action lists.

## Deterministic constraints

`planning.yaml` is the single planning authority. Its conservative operational defaults are four actions per plan and 180 minutes per period; these are safety constraints, not empirical claims about optimal workload. The shared solver checks dataset/stage applicability, action count, per-period workload, duplicates, prerequisites, contraindications, configured conflicts, direct evidence, human-contact flags, automation status, and final-stage prohibition.

Ordering is ordinal only: CRITICAL, HIGH, MEDIUM, LOW. Stable tie-breaking uses evidence completeness, stage urgency, directness, lower equivalent workload, then `action_id`. No randomness, score, probability, or default monitoring action is introduced. Prerequisites are topologically ordered before dependent actions.

## Dataset planning

UCI MAT and POR share a builder but retain distinct Phase 3 policies. S0 cannot use G1/G2, S1 cannot use G2, and S2 never uses G3. Business periods are `CURRENT_PERIOD`, `NEXT_ASSESSMENT`, and `FOLLOW_UP`; no fictional weekly calendar is inferred.

OULAD uses `IMMEDIATE`, `SHORT_TERM`, and `FOLLOW_UP` only when sufficient course time remains. The requested cutoff is retained while the latest validated past prediction anchor is recorded. Plans never extend beyond 100%; less than 10% remaining limits selection to one immediate action and yields PARTIAL when other eligible actions are truncated. FINAL_EVALUATION produces zero interventions.

## Explanation and persistence

Plan explanations contain only selected evidence values and source lineage, policy reason codes, excluded actions, explicit constraints/limitations, and routing metadata. They make no causal or educational-effectiveness claim.

`JsonPlanRepository` stores one immutable versioned JSON document per deterministic plan ID. `PostgresPlanRepository` maps the same complete contract into the existing `recommendation.plan.payload` and `recommendation.action.payload` JSONB columns using inserts only; it neither alters schema nor overwrites legacy rows. Advisor review remains optional.

## Gate boundary

Phase 4 validates software safety, persistence and deterministic replay only. Educational effectiveness, causal evaluation, dashboards, thesis conclusions, and Phase 5 work are outside this phase.
