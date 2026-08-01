# Recommendation V2 architecture specification

## Scope and invariant

Recommendation V2 ranks support actions; it never predicts academic outcome again. The selected CNN-BiLSTM checkpoint is frozen. Phase 1 defines this design only.

```text
Pre-cutoff student data
  -> Frozen CNN-BiLSTM/H1 authority
       -> prediction class
       -> class probabilities
       -> calibrated confidence
       -> uncertainty / seed disagreement
       -> 64-D student_state_embedding
  + ObservedLearningState with evidence mask and lineage
  -> Recommendation Adapter
  -> Neural Action Ranker (Phase 3, expert-supervised)
  -> Constraint Solver
  -> Multi-week Learning Plan
  -> Advisor Review
```

## 1. Frozen prediction adapter

The adapter accepts a versioned `PredictionContext` and `StudentRepresentation`. It loads only the architecture selected by an explicit baseline resolution record, verifies checkpoint/config/architecture hashes, calls `eval()` under inference mode and exposes immutable outputs. For canonical OULAD H1, read `binary_logit`, sigmoid probability, thresholded class, `outcome_logit` softmax, seed ensemble dispersion and `student_state_embedding` `[B,64]`. Calibration must be fitted only from inner/validation data belonging to the locked prediction protocol. Do not use the historical independent ML cross-check as a required V2 input; seed disagreement is the primary ensemble uncertainty signal.

Because H1 includes a direct 32-D tabular residual branch, provenance must state that the 64-D embedding represents the CNN-BiLSTM fused backbone while probabilities reflect both backbone and bounded residual path. The adapter must not detach and recompute the prediction head from an incomplete representation.

## 2. Observed learning state

Allowed fields are cutoff-safe activity level, inactivity streak, assessment progress, grade trend, course progress, weeks remaining and evidence availability mask. Every value carries raw source, transformation version, maximum source timestamp/day and cutoff comparison. Missing is distinct from zero. Sensitive attributes and outcome/post-cutoff fields are rejected before ranking.

## 3. Action catalog

Each immutable catalog version contains `action_id`, category, localized description, workload minutes, applicable stages, required evidence, prerequisites, contraindications, incompatible actions, human-review requirement, availability dates/resources and safety notes. Existing seven actions are candidates for review, not automatically approved catalog entries. Catalog validation precedes label collection so reviewers score identical candidates.

## 4. Recommendation adapter

The adapter concatenates normalized frozen embedding, calibrated class probabilities, scalar/masked uncertainty features and normalized observed state plus evidence mask. It may project these inputs into a recommendation representation, but receives no academic target and has no outcome-prediction loss. Frozen inputs are detached; gradients stop at the prediction boundary. Adapter/version lineage includes model/checkpoint, calibration, feature contract, action catalog and cutoff.

## 5. Neural Action Ranker

One shared scoring network evaluates `(student recommendation representation, candidate action representation)` and returns an action relevance score. Training uses only real expert ratings, grouped student splits and candidate exposure masks. It does not generate pseudo-labels, predict risk or change the CNN-BiLSTM. Metrics: NDCG@K, Precision@K, Recall@K, coverage and safety slices versus the governed rule baseline.

## 6. Constraint Solver

Apply hard constraints after scoring: catalog/stage applicability, required evidence, prerequisites, contraindications, incompatibilities, unique action IDs, maximum actions and workload. Uncertainty can reduce automated scope, require review or trigger abstention, never fabricate evidence. Solver returns selected/rejected candidates with reason codes and deterministic tie-breaking. If no safe set exists, return abstention/escalation rather than a default action.

## 7. Learning Plan Builder

Convert selected actions into weeks without changing their rank semantics. Each item carries goal, week/timing, workload, rationale, supporting evidence, success criterion and review point. Total weekly workload remains capped. `PROGRESS_MONITORING` is not automatically inserted; it competes or is added only by an explicit catalog/solver rule with lineage.

## 8. Advisor review

Persist one of `APPROVED`, `MODIFIED`, `REJECTED`, `NEEDS_MORE_EVIDENCE`, reviewer identity, timestamp, reason and complete before/after diff. Modification creates a new plan revision linked to the original. No high-impact plan is presented as approved before human review when the catalog or uncertainty policy requires it.

## Required safety and validation

- Prediction/probability invariance, deterministic embedding shape and checkpoint immutability.
- No post-cutoff/sensitive fields; complete lineage and safe missing-evidence behavior.
- Unique catalog IDs, valid workload/stages/prerequisites and no undefined dependency.
- Student-group split, reproducibility and no label/prediction leakage.
- Workload/action caps, duplicate/prerequisite enforcement and correct abstention.
- Deterministic end-to-end replay with model, policy, catalog, solver, plan and review versions.

## Phase 1 blocker

Implementation must not start until an architecture decision record selects either historical F2 `cnn_bilstm_oulad` or canonical V3 H1/cutoff authority and synchronizes the primary CLI, config registry, checkpoint manifest and recommendation lineage.
