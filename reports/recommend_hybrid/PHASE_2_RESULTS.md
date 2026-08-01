# recommend_hybrid Phase 2 results

## Result

`PHASE_2_PASS`. The dedicated validator reports `RECOMMEND_HYBRID_PHASE2_FOUNDATION_PASS`.

## Prediction adapter

Authority is `RECOMMEND_HYBRID_MODEL_AUTHORITY`. Phase 1 revalidated all 30 canonical checkpoints; the Phase 2 invariance test loaded the five shared `MIDDLE_50`, fold-0 seed checkpoints. Direct and adapted logits, probabilities, classes and embeddings are exactly equal on the same CPU/float32 path (maximum difference 0, tolerance 0). Checkpoint and parameter hashes are identical before/after. Embeddings are 64-D and 32-D; deterministic replay passes.

## Safety and candidates

All five canonical stages are typed. Observed state enforces `event_day < cutoff_day`, reports zero post-cutoff and sensitive-feature violations, keeps missing evidence as `None`, and requires per-feature lineage. Ten active actions pass ID, stage, workload, evidence and dependency validation. Candidate generation contains no score/rank and produces zero final-stage intervention candidates.

## Expert pipeline

Sixty real-data, blinded `MIDDLE_50` pilot cases were exported with blank action/case-review templates for two planned reviewers. Exact probability is banded under the approved blinding protocol. Actual reviewers: 0; scored cases: 0; ratings: 0; fabricated labels: 0. Invalid-score and duplicate import checks pass. `expert_status=PENDING_REAL_EXPERT_LABELS`; `phase3_training_status=BLOCKED`.

## Boundary

No `HybridActionRanker`, constraint solver, learning-plan builder, production API/database integration, prediction training or Phase 3 work was performed.
