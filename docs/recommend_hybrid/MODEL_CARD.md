# Model card: Hybrid CNN-BiLSTM Learning Support Recommender

## System identity and purpose

System name: **Hybrid CNN-BiLSTM Learning Support Recommender**. Module: `recommend_hybrid`. Authority: `RECOMMEND_HYBRID_MODEL_AUTHORITY`. The system converts a frozen academic-risk prediction and cutoff-safe learning evidence into a constrained support plan. Its purpose is decision support and auditable plan generation, not autonomous educational decision-making.

## Supported datasets and temporal scope

- UCI `student_mat` and `student_por`: S0 (no grade), S1 (G1 only), and S2 (G1/G2); G3 is prohibited.
- OULAD: EARLY_20, EARLY_35, MIDDLE_50, LATE_75, and FINAL_EVALUATION. Arbitrary requests from 20–99% use the latest validated past anchor. Requests before 20% abstain; final is evaluation-only.

## Prediction dependency

The prediction authority is the frozen hybrid CNN-BiLSTM family with architecture hash `df5cd885b96e5cea4b840bfc5ca59c08c095f5887df8dd8dcef738edfe8bf70e`, 160,492 parameters and five fixed seeds. The canonical checkpoint manifest is immutable. CNN-BiLSTM is the deep-learning prediction component; recommendation uses its class/probability/uncertainty context together with observed business evidence.

## Recommendation logic

The recommendation component is a deterministic evidence-based policy, not a trained neural ranker. Dataset-specific rules establish eligibility and ordinal priority; the Phase 4 selector and constraint solver enforce dataset/stage applicability, evidence, prerequisites, contraindications, conflicts, human-contact flags, four-action maximum and 180-minute per-period maximum. No action probability, pseudo-label, random selection, auxiliary ML model, or effectiveness estimator is used.

## Inputs and outputs

Inputs are pseudonymous student/course keys, frozen prediction context, canonical stage or requested cutoff, observed pre-cutoff activity/assessment evidence, missingness, source lineage and version authority. Sensitive attributes, future outcomes, test labels and post-cutoff evidence are prohibited. Outputs are `LearningPlan` contracts with FULL, PARTIAL, ABSTAIN or EVALUATION_ONLY status, selected actions, periods, workloads, reason codes, evidence, limitations, authority lineage and deterministic plan ID.

## Safety, abstention and explanations

ABSTAIN produces no actions when evidence/anchor is insufficient, contradictory, or uncertainty exceeds the locked policy. EVALUATION_ONLY also produces no interventions. Uncertainty can only reduce priority or automation. Every selected action requires direct available evidence and lineage. Explanations state observed conditions and policy rationale but never promise grade improvement or causal benefit.

## Evaluation methodology and result summary

Final technical evaluation uses 260 deterministic records sampled without outcome labels from locked canonical hybrid OOF/seed predictions and raw pre-cutoff features: 60 MAT, 60 POR, 100 canonical OULAD-anchor records and 40 OULAD inter-stage routing records. It is a technical evaluation sample, not a population-effectiveness estimate. All safety/constraint violation counts are 0; evidence support and explanation lineage are 100%; deterministic replay is 100%. Actionable coverage is 92.08% of 240 intervention-eligible records and abstention is 7.92%. Student-level percentile bootstrap uses 1,000 replicates and fixed seed 20260801.

## Action concentration

`PROGRESS_MONITORING` appears in 78.28% of actionable plans and is the most frequent action (27.90% of selected actions). Audit confirms it is not inserted by the selector as a default and every occurrence has supporting evidence. The concentration follows the Phase 3 monitoring eligibility rule, whose LOW severity accepts broadly available indicators. This is reported as a limitation; Phase 5 does not modify the locked policy.

## Known limitations and prohibited uses

There are no real expert ratings, user-acceptance measurements, deployment outcomes, causal effects, or action-optimality labels. The system must not be described as expert validated, optimal, recommendation-accurate, or proven to improve grades. It must not be used for punitive decisions, automated withdrawal, high-stakes allocation without human review, sensitive-attribute profiling, or recommendations outside supported datasets/stages. Metrics such as Precision@K, Recall@K, NDCG and MRR are not meaningful without valid relevance labels.

## Claim boundary and version lineage

Technical validity, temporal safety, evidence linkage, deterministic replay and supported routing are established within the evaluation boundary. Educational effectiveness, expert validation, user acceptance and causal impact are not supported. Version lineage is Phase 1 authority → Phase 2 frozen adapter/contracts → Phase 3 evidence policy → Phase 4 constrained planning → Phase 5 scientific evaluation. Reproduction requires the release manifest, exact policy/planning hashes, canonical prediction artefacts, fixed bootstrap seed and validator `RECOMMEND_HYBRID_PHASE5_FINAL_PASS`.
