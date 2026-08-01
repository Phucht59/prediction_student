# Hybrid CNN-BiLSTM Learning Support Recommender

## Scientific identity

- Module: `recommend_hybrid`
- Vietnamese name: **Mô hình khuyến nghị hỗ trợ học tập dựa trên CNN-BiLSTM hybrid**
- Architecture family: `HYBRID_CNN_BILSTM_RECOMMENDER`
- Prediction backbone: `FROZEN_HYBRID_CNN_BILSTM`
- Recommendation component: `EVIDENCE_BASED_KNOWLEDGE_RECOMMENDATION_POLICY`
- Neural action ranker: disabled
- Recommendation training: not applicable
- Prediction authority: `RECOMMEND_HYBRID_MODEL_AUTHORITY`

## Current architecture

```text
RECOMMEND_HYBRID
├── RecommendHybridUCI
│   ├── student_mat policy/config
│   ├── student_por policy/config
│   ├── G1/G2 availability stage router
│   ├── UCI evidence severity
│   └── UCI-only evidence-action policy
├── RecommendHybridOULAD
│   ├── arbitrary requested-cutoff router
│   ├── past-only validated prediction anchor
│   ├── requested-cutoff observed evidence
│   └── OULAD-only evidence-action policy
└── Shared core
    ├── typed contracts and ordinal priorities
    ├── uncertainty reduction and abstention
    ├── evidence lineage and explanations
    └── deterministic scientific validation
```

Both branches consume class/probabilities/uncertainty from the frozen CNN-BiLSTM authority. They share output contracts and safety controls, but never share business rules, severity thresholds, or dataset-only actions. The policy returns eligible considerations with `CRITICAL/HIGH/MEDIUM/LOW`; it returns no relevance probability, learned action score, Top-K plan, or educational-effectiveness claim.

## Representation boundary

The existing 64-D student-state and 32-D tabular-expert representations remain in lineage for reproducibility and future research. Phase 3 decision modules do not read either representation. Without independent action labels, embeddings cannot determine eligibility or priority.

## Routing and information safety

UCI stage is determined only by actual G1/G2 availability: S0 has neither, S1 has G1, and S2 has G1/G2. G3 is forbidden. MAT and POR use separate configs.

OULAD routes an arbitrary request to the nearest validated anchor in the past. Observed evidence may extend to the actual requested cutoff, but prediction validation is claimed only at the anchor. Requests before 20% abstain; 100% is evaluation-only. Every OULAD event must be strictly earlier than the requested cutoff.

## Historical phase boundary

Phase 1 froze prediction authority and originally documented a possible future neural ranker. Phase 2 created an optional expert-review pipeline. The user-approved Phase 3 decision supersedes that recommendation-training path: expert review is now a future extension, not a dependency. Frozen prediction/checkpoint authority and all Phase 1/2 historical reports remain unchanged.

Phase 3 ends at eligible actions and ordinal priority. Final selection, workload constraints, multi-week plans, API/database integration and outcome evaluation remain outside this phase.
