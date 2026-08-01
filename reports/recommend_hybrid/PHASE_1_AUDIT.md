# recommend_hybrid Phase 1 authority audit

## Result

`PHASE_1_PASS`. The dedicated authority validator reports `RECOMMEND_HYBRID_PHASE1_AUTHORITY_PASS`; all required canonical stage checkpoints exist and no checkpoint or prediction baseline changed.

## Approved authority

The official scientific system is **Hybrid CNN-BiLSTM Learning Support Recommender** (`recommend_hybrid`), architecture family `HYBRID_CNN_BILSTM_RECOMMENDER`, authority `RECOMMEND_HYBRID_MODEL_AUTHORITY`. Its prediction backbone is `FROZEN_HYBRID_CNN_BILSTM`; no separate prediction model is allowed.

Locked architecture: SHA-256 `df5cd885b96e5cea4b840bfc5ca59c08c095f5887df8dd8dcef738edfe8bf70e`, 160,492 parameters, 64-D student-state embedding and 32-D tabular-expert embedding. Recommendation input is both embeddings plus prediction probabilities, uncertainty and observed pre-cutoff learning state.

## Stage authority

| Stage | Authority use | Checkpoint role |
|---|---|---|
| `EARLY_20` | early screening; recommend only with sufficient evidence | shared intervention checkpoint |
| `EARLY_35` | early recommendation | shared intervention checkpoint |
| `MIDDLE_50` | **PRIMARY_RECOMMENDATION_STAGE** | shared intervention checkpoint |
| `LATE_75` | late intervention or advisor escalation | shared intervention checkpoint |
| `FINAL_EVALUATION` | evaluation only; never generate a new plan | dedicated endpoint checkpoint |

## Canonical checkpoint inventory

The repository contains 15 shared-stage files and 15 dedicated evaluation files: three outer folds × five seeds for each role. Fold identity is supported by the role-specific training authority and the checkpoint's training-policy hash; seed, architecture hash and parameter count are stored inside every checkpoint payload. Stage scope is supported by the training authority and canonical benchmark code, not inferred from filenames alone.

The manifest expands the 15 shared files over four intervention stages and the 15 endpoint files over `FINAL_EVALUATION`, producing 75 unique `(stage, outer_fold, seed)` mappings. Expected 30, found 30, missing 0, invalid 0. Set status: `COMPLETE_MULTI_STAGE_CHECKPOINT_SET`.

## Output and representation contract

The frozen model exposes binary risk logit/probability, auxiliary outcome probabilities, class/threshold decision and two representations. `student_state_embedding` is the 64-D fused CNN-BiLSTM/aggregate/static representation before the main head. The 32-D tabular-expert representation contributes through the bounded residual path and is therefore included in the recommendation input. Ensemble dispersion across the five seeds supplies uncertainty; any calibrated confidence must retain its calibration lineage.

Extraction is feasible without changing prediction results: read existing forward outputs under `eval()` and inference mode, detach them at the adapter boundary, and prohibit gradients into the backbone. Phase 2 must prove exact checkpoint/state immutability and prediction/probability invariance before use.

## Current recommendation inventory

The current service retrieves stored plans; it does not generate new ones. Historical rule systems provide useful lineage, workload, duplicate, abstention and advisor-review patterns but are not the new ranking authority. The old independent weak-label neural recommender is excluded because it predicts support risk separately rather than ranking actions from the frozen hybrid representation.

KEEP retrieval, version lineage, cutoff/sensitive-feature protection, abstention, workload/action caps, duplicate protection and advisor review. REFACTOR contracts, observed state, candidate catalog, prerequisite handling and review states. REPLACE rule-based action selection. REMOVE the independent weak-label prediction path. The new action ranker, constraint solver and plan builder remain MISSING and are not implemented in Phase 1.

## Expert data

Existing packages contain templates and prepared cases, not real labels: reviewers 0, scored cases 0, plan/action metrics null and inter-rater agreement pending. `expert_status = PENDING_REAL_EXPERT_LABELS`; `training_status = LOCKED_UNTIL_REAL_EXPERT_LABELS`. No pseudo-label is permitted.

## Provenance and legacy classification

The approved source chain is recorded only as provenance: `FINAL_THESIS_MODEL_AUTHORITY` → historical source alias `UNIFIED_CANONICAL_BENCHMARK_V3` → historical model alias `H1_TABULAR_RESIDUAL_EXPERT`.

- Historical recommendation evidence from source aliases V6/V6.2 and stage alias F2: `REFERENCE_ONLY`, not canonical or training authority.
- Persisted database plans from source alias V5.2: `LEGACY_EVIDENCE_ONLY`.
- Primary 65-checkpoint legacy validator: `LEGACY_COMPATIBILITY_ONLY`.
- Canonical checkpoint source: `RECOMMEND_HYBRID_SOURCE_OF_TRUTH`.

The seven report-checksum failures in the legacy validator are `PRE_EXISTING_LEGACY_VALIDATOR_FAILURE` and `NOT_RECOMMEND_HYBRID_AUTHORITY`. No old artefact is deleted, renamed or substituted; paths containing historical aliases remain only as provenance.

## Phase 1 boundary

This resolution creates authority/config/manifest/validator and documentation only. It does not implement `HybridPredictionAdapter`, `HybridActionRanker`, an action catalog, expert pipeline, API/schema changes or inference over a dataset. No checkpoint, prediction result or database row is modified.
