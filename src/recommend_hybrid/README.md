# Recommendation source

## Production authority

Use `src/recommend_hybrid/final/` for the validated recommendation system.
The former conditional-action ranker that previously occupied the `final/`
namespace has been replaced by the released five-EBM Recommendation system.

```python
from src.recommend_hybrid import RecommendationPipeline
from src.recommend_hybrid.final import FiveEBMRanker, RouteStatus
```

The validated flow is:

1. frozen Hybrid CNN-BiLSTM risk prediction;
2. risk/evidence routing;
3. hard feasibility filtering;
4. five action-specific EBM relevance models;
5. valid-action ranking;
6. safety routing;
7. evidence-grounded recommendation and learning plan;
8. plausibility simulation reported only as **model-implied risk delta**.

The four public route statuses are `RECOMMEND`, `INSUFFICIENT_EVIDENCE`,
`HUMAN_REVIEW`, and `NO_FEASIBLE_ACTION`.

## Final held-out evidence

The one-shot `PANEL_B_FINAL_HELDOUT` benchmark used 150 cases and 557 real
external Gemini review records. The frozen five-EBM ranker achieved:

| Metric | Final result |
|---|---:|
| NDCG@3 | 0.9526603067902532 |
| Top-1 agreement | 0.92 |
| Precision@1 | 0.9733333333333334 |
| MRR | 0.9855555555555556 |
| Recall@3 | 0.8247777777777778 |
| Pairwise accuracy | 0.8353552859618717 |
| Invalid-action rate | 0.0 |

The frozen action+stage baseline reached NDCG@3 `0.8275943281032121`.
The paired-bootstrap mean delta was `+0.12466302441561493`, with 95% CI
`[0.09508467988207753, 0.15361541252930452]`.

Canonical evidence is under `artifacts/recommend_hybrid/final/` and the public
scientific reports are under `reports/recommend_hybrid/final/`.

## Frozen lineage and legacy code

`src/recommend_hybrid/explainable_v2/` remains byte-preserved as the scientific
implementation lineage that produced the final release. It is not the public
production namespace and is intentionally not renamed after Panel B because its
location participates in the historical audit trail.

Other older recommendation, causal, policy, and strategy modules remain only
for provenance or non-authoritative research utilities. They must not be
presented as the final recommendation model. In particular, the released
simulator makes no causal effectiveness claim.
