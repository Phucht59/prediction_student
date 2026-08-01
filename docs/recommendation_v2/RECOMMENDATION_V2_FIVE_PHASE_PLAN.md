# Recommendation V2 five-phase plan

Every phase requires explicit user approval and a recorded gate. Failure or blocked status cannot automatically advance.

## Phase 1 — Audit, baseline lock and design

Deliver repository/model/recommendation/database audit, `BASELINE_LOCK.json`, inventory, architecture, data contracts, expert protocol and validation plan. Gate: singular prediction authority, verified checkpoints/hashes, extraction point, honest expert status, valid artefacts and no production/model change. Current gate: `PHASE_1_BLOCKED` pending OULAD authority resolution.

## Phase 2 — Adapter, contracts, observed state, catalog and label pipeline

After explicit approval, implement extraction-only frozen CNN-BiLSTM adapter, typed contracts, cutoff-safe observed state, versioned action catalog and real expert-label import/export/validation. Gate: prediction/probability invariance, checkpoint immutability, deterministic embedding, no post-cutoff/sensitive data, catalog integrity and an operational label pipeline. Do not train the ranker.

## Phase 3 — Real labels and Neural Action Ranker

Import real expert ratings, freeze student-grouped splits, train only the recommendation adapter/ranker and evaluate against governed rules. Gate: label quality/agreement, no student leakage, reproducibility, NDCG@K/Precision@K/Recall@K thresholds, safety review and frozen predictor unchanged.

## Phase 4 — Constraint solver, plan builder and integration

Implement hard constraints, abstention/escalation, explanations, multi-week plan builder, version lineage, API and database integration. Gate: workload/action/duplicate/prerequisite tests, deterministic replay, safe failure handling, migration validation and advisor-review workflow. No silent production cutover.

## Phase 5 — Expert evaluation and final release

Run real expert evaluation, technical and leakage audits, model card, thesis updates and release validation. Gate: agreed expert metrics, causal-claim boundary, database/API reconciliation, full checksum manifest, rollback evidence and explicit release approval.

The next allowed phase is Phase 2 only after the user resolves the Phase 1 blocker and explicitly approves continuation.
