# Recommendation system

## Production authority

The released recommendation model lives in `src/recommend_hybrid/final/`.
The previous conditional-action recommender is no longer the production authority.

```python
from src.recommend_hybrid import RecommendationPipeline
from src.recommend_hybrid.final import FiveEBMRanker, RouteStatus
```

Validated flow:

1. frozen Hybrid CNN-BiLSTM risk prediction;
2. risk/evidence routing;
3. hard feasibility filtering;
4. five action-specific EBM relevance models;
5. valid-action ranking;
6. four-status safety routing;
7. evidence-grounded recommendation and learning plan;
8. plausibility simulation reported only as **model-implied risk delta**.

The public statuses are `RECOMMEND`, `INSUFFICIENT_EVIDENCE`, `HUMAN_REVIEW`, and
`NO_FEASIBLE_ACTION`.

## Final evidence

`PANEL_B_FINAL_HELDOUT` was evaluated exactly once on 150 cases with 557 real external
Gemini review records. The frozen ranker reached NDCG@3 `0.9526603067902532`; the
frozen action+stage baseline reached `0.8275943281032121`. The paired-bootstrap mean
difference was `+0.12466302441561493`, 95% CI
`[0.09508467988207753, 0.15361541252930452]`, with invalid-action rate `0.0`.

Release evidence: `artifacts/recommend_hybrid/final/`.
Final reports: `reports/recommend_hybrid/final/`.
Final configuration: `configs/recommend_hybrid/final/`.

The immutable scientific lineage is retained on Git branch `Module_recomend`, release
commit `17b519b22e8b69c875d27547d097e6d3b76bc404`. No Panel-B rerun or post-heldout
model tuning is permitted.
