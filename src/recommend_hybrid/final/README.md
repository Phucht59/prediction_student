# Final recommendation module

This directory is the production implementation of the released recommendation system.
It replaces the previous conditional-action recommender that formerly occupied the
`src/recommend_hybrid/final/` namespace.

Pipeline:

```text
Frozen Hybrid CNN-BiLSTM risk model
        -> risk / evidence routing
        -> hard feasibility filter
        -> five frozen action-specific EBMs
        -> valid-action ranking
        -> safety router
        -> recommendation + explanation + learning plan
        -> plausibility simulator
```

Public router statuses are exactly:

- `RECOMMEND`
- `INSUFFICIENT_EVIDENCE`
- `HUMAN_REVIEW`
- `NO_FEASIBLE_ACTION`

The frozen EBM native relevance scale is ordinal `[0, 3]`. The public score uses one
normalization only: `clip(native_prediction / 3, 0, 1)`.

The simulator reports **model-implied risk delta** only and makes no causal treatment
claim.

## Final held-out evidence

The one-shot Panel-B benchmark used 150 held-out cases and 557 real external Gemini
review records. The frozen five-EBM ranker achieved NDCG@3 `0.9526603067902532`,
compared with `0.8275943281032121` for the frozen action+stage baseline. The paired
bootstrap mean NDCG difference is `+0.12466302441561493`, with 95% CI
`[0.09508467988207753, 0.15361541252930452]`. Invalid-action rate is `0.0`.

Canonical evidence is in `artifacts/recommend_hybrid/final/` and final reports are in
`reports/recommend_hybrid/final/`.

The immutable scientific source release remains available on branch `Module_recomend`
at commit `17b519b22e8b69c875d27547d097e6d3b76bc404`. The files here are the cleaned
production namespace derived from that frozen release; Panel B is never rerun or used
for post-heldout tuning.
