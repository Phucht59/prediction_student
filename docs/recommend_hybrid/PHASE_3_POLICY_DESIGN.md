# Phase 3 dual-dataset evidence policy design

## Purpose

Phase 3 implements an `EVIDENCE_BASED_KNOWLEDGE_RECOMMENDATION_POLICY`. It is deterministic policy software, not a supervised recommendation model. It consumes frozen CNN-BiLSTM prediction context and direct pre-cutoff learning evidence, then returns action eligibility, ordinal priority, automation status and evidence-faithful explanations.

## Shared controls

The shared layer defines immutable requests/results, evidence availability/severity, ordinal priority, uncertainty disposition, abstention and explanations. Direct evidence is mandatory before frozen risk class/probability can adjust an action priority. The predeclared probability band is contextual only and is never called action suitability. Risk alone cannot trigger an action. Uncertainty can cap priority, produce `PARTIAL`, or force `ABSTAIN`; it cannot increase priority or automation.

Eligibility is evaluated before priority. Ineligible actions always have `NOT_APPLICABLE`. `ABSTAIN` exposes no eligible action and never silently becomes progress monitoring. Contradictory activity/inactivity evidence is an explicit abstention condition.

## Dataset branches

`RecommendHybridUCI` supports `student_mat` and `student_por` with separate config hashes and thresholds. `RecommendHybridOULAD` supports arbitrary cutoffs using past-only validated anchors. Each branch owns its evidence definitions and action rules; only contracts, status vocabulary, uncertainty/abstention mechanics, reason lineage and validation framework are shared.

## Configuration and provenance

All severity thresholds, stages, action triggers, priority caps, uncertainty thresholds and abstention rules are declared in four YAML files. Thresholds are predeclared from direct variable domains/information policy, never selected after test outcomes. Config provenance records source policy hashes and confirms no test/outer label use.

## Excluded behavior

Phase 3 does not use embeddings for decisions, train a neural ranker, create suitability probabilities, choose Top-K, optimize total workload, build a multi-week plan, write production data, expose an API, or estimate educational effectiveness.
