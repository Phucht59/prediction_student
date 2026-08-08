# Final recommendation module

This directory is the **production-facing recommendation API**.

The previous conditional-action recommendation implementation that occupied this
path has been replaced by the scientifically released risk-guided system built
from five action-specific EBM rankers.

Production flow:

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

The EBM native relevance scale is ordinal `[0, 3]`; the public score uses one
adapter only: `clip(native_prediction / 3, 0, 1)`.

The simulator reports **model-implied risk delta** only. It does not claim a
causal treatment effect.

## Scientific authority

Final held-out Panel-B results:

- 150/150 cases;
- 557 real external Gemini review records;
- NDCG@3 `0.9526603067902532`;
- action+stage baseline NDCG@3 `0.8275943281032121`;
- paired bootstrap NDCG delta `+0.12466302441561493`;
- 95% CI `[0.09508467988207753, 0.15361541252930452]`;
- Top-1 agreement `0.92`;
- Precision@1 `0.9733333333333334`;
- invalid-action rate `0.0`.

Canonical release evidence is exposed under
`artifacts/recommend_hybrid/final/` and canonical reports under
`reports/recommend_hybrid/final/`.

## Why the old `explainable_v2` implementation path remains

The old versioned path is retained **only as frozen scientific lineage**. Moving
or rewriting those already-audited files after the one-shot Panel-B evaluation
would make evidence reproduction harder and could invalidate path-based audit
assumptions. Application code must use `src.recommend_hybrid.final`; the
versioned path is not the public production namespace.
