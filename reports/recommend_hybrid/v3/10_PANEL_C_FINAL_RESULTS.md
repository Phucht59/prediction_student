# 10 — Panel C final held-out results

**STATUS: EVALUATED**

This is the only official V3 held-out claim. Development weak-label metrics are not held-out.

| Model | cases | NDCG@3 | P@1 | MRR | R@3 | pairwise | invalid |
|---|---:|---:|---:|---:|---:|---:|---:|
| Recommendation V | 632 | 0.88785 | 0.99206 | 0.99603 | 0.79947 | 0.53849 | 0.0 |
| B0 action+stage | 632 | 0.81889 | 0.99365 | 0.99683 | 0.78981 | 0.25377 | 0.0 |
| B1 rule score | 632 | 0.86649 | 0.99683 | 0.99841 | 0.80357 | 0.45402 | 0.0 |

Exact-best Top-1 agreement: 0.40711462450592883

Bootstrap Recommendation V minus B1 NDCG@3: mean=0.02131, 95% CI [0.01440, 0.02815], P(diff>0)=1.0000, iterations=2000, seed=2026.

Pipeline system rates: {
  "n_cases": 632,
  "invalid_action_rate": 0.0,
  "recommendation_coverage": 0.14873417721518986,
  "HUMAN_REVIEW_rate": 0.27689873417721517,
  "INSUFFICIENT_EVIDENCE_rate": 0.5743670886075949,
  "NO_FEASIBLE_ACTION_rate": 0.0,
  "route_counts": {
    "RECOMMEND": 94,
    "INSUFFICIENT_EVIDENCE": 363,
    "HUMAN_REVIEW": 175,
    "NO_FEASIBLE_ACTION": 0
  },
  "top1_action_distribution": {
    "RECOVER_ENGAGEMENT": 111,
    "QUIZ_RETRIEVAL_PRACTICE": 64,
    "TARGETED_CONTENT_REVIEW": 36,
    "STUDY_REGULARITY": 31,
    "ASSESSMENT_COMPLETION": 27
  },
  "unique_top1_actions": 5
}

Historical Panel B is previous recommendation release held-out evidence and was not used for V3 tuning or this evaluation.
