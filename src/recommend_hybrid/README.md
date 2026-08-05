# Recommendation source

## Final conditional ranker

Use `src/recommend_hybrid/final/` for the conditional recommendation component.
`ConditionalHybridActionRanker` ranks policy-authorized scientific actions by
identity using output from the integrated conditional action head over the
frozen residual CNN–BiLSTM representation.

```python
from src.recommend_hybrid.final import ConditionalHybridActionRanker
```

The caller must explicitly provide external eligibility and integrated-head
output. The final API fails closed when eligibility or model scores are absent.
The fixed model action order is defined in `final/actions.py`; policy catalog
aliases are mapped to those trained action identities before ranking.

The final ranker now has a four-stage contract:

- `EARLY_20`
- `EARLY_35`
- `MIDDLE_50`
- `LATE_75`

Legacy three-stage threshold artifacts remain replayable for the first three
stages, but they cannot authorize `LATE_75`. A new local training/evaluation run
must produce a four-value stage threshold vector before late-stage issuance.

## Stage-aware causal evidence

Use `src/recommend_hybrid/causal/` for the new observational target-trial
subsystem. It evaluates an action at the same landmark that produced the
recommendation:

| Ranking stage | Baseline information | Observed treatment window |
|---|---:|---:|
| `EARLY_20` | 0–20% | after 20% through 35% |
| `EARLY_35` | 0–35% | after 35% through 50% |
| `MIDDLE_50` | 0–50% | after 50% through 75% |
| `LATE_75` | 0–75% | after 75% through course end |

The subsystem provides:

- cross-fitted AIPW doubly robust estimation;
- out-of-fold CATE estimates;
- propensity overlap trimming;
- weighted standardized mean-difference diagnostics;
- effective sample-size gates;
- student-cluster percentile bootstrap;
- train-fitted treatment definitions for all five canonical actions;
- fail-closed `CAUSAL_EVIDENCE_NOT_IDENTIFIABLE` status;
- latest-valid-recommendation-wins lifecycle logic.

Because the recommender has not been deployed, treatment means **observed
post-landmark behaviour consistent with the action**, not confirmed receipt or
compliance with a displayed recommendation.

## Frozen Hybrid imbalance evidence

`src/recommend_hybrid/causal/imbalance.py` compares:

- `none`;
- `class_weight`;
- `smote`;
- `adasyn`.

SMOTE and ADASYN are applied only to frozen Hybrid training embeddings.
Validation is used only to choose the operational threshold and test is used
only for final metrics. The canonical frozen prediction checkpoint is never
automatically replaced by this sensitivity study.

## Scientific boundary

Already supported by the existing release:

- conditional action ranking evaluated offline;
- cutoff-safe evidence linkage;
- deterministic action identity mapping;
- fail-closed authorization and abstention.

Supported only after the new local artifacts pass the causal release gate:

- observational stage-action effect estimates under measured-confounding,
  positivity, consistency, and model assumptions;
- estimated benefit heterogeneity for rows inside the overlap population.

Still unsupported:

- randomized causal effectiveness;
- proof that displaying a recommendation improves grades;
- guaranteed grade improvement;
- expert validation;
- user acceptance;
- production runtime authorization.

The legacy `pipeline.py`, `uci/`, `oulad/`, and `common/` modules are retained
for compatibility and constraint/policy utilities. They are not the validated
final action-ranking entry point and must not be presented as having 93.74%
end-to-end recommendation accuracy.
