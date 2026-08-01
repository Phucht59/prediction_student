# recommend_hybrid five-phase plan

Each phase has an explicit gate and requires user approval. No phase advances automatically.

## Phase 1 — Audit and frozen authority

Lock `RECOMMEND_HYBRID_MODEL_AUTHORITY`, canonical checkpoint manifest, stage policy, architecture, embeddings, current inventory, expert status and dedicated validation. Gate: all real checkpoint paths/hashes/payloads and 75 stage/fold/seed mappings pass; no prediction or database change.

## Phase 2 — Frozen adapter and data foundation

Implement `HybridPredictionAdapter`, contracts, cutoff-safe observed state, versioned action catalog and real expert-label import/export validation. Gate: checkpoint immutability, exact prediction/probability invariance, deterministic embeddings, no post-cutoff/sensitive data and catalog integrity. No ranker training.

## Phase 3 — Real labels and action ranking

Import real ratings, freeze student-grouped splits, train only `HybridActionEncoder`/`HybridActionRanker`, and compare against the governed rule baseline using NDCG@K, Precision@K and Recall@K. Gate: label quality/agreement, reproducibility, leakage and safety thresholds.

## Phase 4 — Constraints, plan building and integration

Implement `HybridConstraintSolver`, `HybridLearningPlanBuilder`, explanation, abstention, advisor review, API and database integration. Gate: workload/action/duplicate/prerequisite/incompatibility tests, deterministic replay, migrations and rollback.

## Phase 5 — Expert evaluation and release

Complete real expert evaluation, technical/leakage audit, model card, thesis material and final release validation. Gate: agreed expert metrics, non-causal claim boundary, complete version/checksum lineage and explicit release approval.

Phase 2 is allowed only after Phase 1 authority validation reports `PHASE_1_PASS` and the user explicitly approves it.
