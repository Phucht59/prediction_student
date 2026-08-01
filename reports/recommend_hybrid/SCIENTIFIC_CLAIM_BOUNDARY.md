# Scientific claim boundary

Version: `recommend_hybrid_scientific_claim_boundary_v1`

The release is evaluated for deterministic technical and policy correctness. It has no real expert ratings, user study, deployed intervention outcomes, action-optimality ground truth, or causal identification design. `SUPPORTED` therefore means supported only within the stated technical evaluation boundary.

| Claim | Status | Evidence and boundary |
|---|---|---|
| Prediction model unchanged | SUPPORTED | Locked architecture hash, parameter count, checkpoint set and Phase 2/4 invariance artefacts match. |
| No post-cutoff leakage | SUPPORTED | 0 post-cutoff violations in 260 locked-prediction evaluation records and targeted cutoff tests. |
| Recommendations evidence-linked | SUPPORTED | Evidence support and explanation-lineage completeness are both 100%; unsupported-reason rate is 0. |
| Supports UCI and OULAD | SUPPORTED | MAT, POR and all canonical OULAD stages execute through dataset-isolated policies and planners. |
| Supports arbitrary OULAD request cutoff | SUPPORTED | Inter-stage 25, 36, 63 and 76 route only to the latest validated past anchor; future-anchor violations are 0. |
| Deterministic replay for identical versioned input | SUPPORTED | Plan and replay hash match rates are 100% on the evaluation set. |
| Policy response is technically safe under declared constraints | SUPPORTED | All declared temporal, dataset, workload, action-cap, duplicate and dependency violation counts are 0. |
| Recommendations are educationally appropriate | PARTIALLY_SUPPORTED | Rules are evidence-linked and internally consistent, but no real expert or user evaluation exists. |
| Recommendations are optimal | NOT_SUPPORTED | No action-optimality ground truth or trained ranker exists. |
| Recommendations improve grades | NOT_SUPPORTED | No intervention outcomes or causal comparison exist. |
| Expert validated | NOT_SUPPORTED | Real reviewer count and real rating count are 0. |
| User accepted | NOT_SUPPORTED | No user study or acceptance measurement exists. |
| Causal effect established | NOT_SUPPORTED | No randomized, quasi-experimental, or causal identification study exists. |
| Recommendation Accuracy / Precision@K / Recall@K / NDCG / MRR | NOT_SUPPORTED | No valid action-relevance ground truth exists, so these metrics are intentionally not reported. |

These statuses must not be upgraded without new evidence collected under a preregistered expert, user, or intervention-evaluation protocol.
