# Cross-Fitted Recommendation Evidence

## Decision

No group remains unused inside the active Phase8 recommendation case universe: all 86 student/groups occur in the feature, weak-label, EBM, threshold/safety, and development-metric scopes. Raw OULAD contains additional students, but they do not have the current action-level relevance/evaluation contract and are not treated as a valid untouched recommendation holdout.

The 121 previously designated final-inference rows were excluded from this evidence. No outcome/relevance values for those rows were opened.

## Evaluation

- scope: 179 genuine Hybrid OOF queries, 50 student/groups
- outer folds: 3, group-disjoint
- inner label-model fit: group-safe nested cross-fitting
- EBM fit: outer-train only, fixed existing parameters, no HPO
- threshold/safety tuning: not performed; no policy claims are made
- primary evidence: cross-validated weak-supervision ranking evidence

Pooled metrics:

```json
{
  "query_count": 179,
  "positive_query_count": 179,
  "ndcg_at_3": 0.9535264211085375,
  "precision_at_1": 0.9776536312849162,
  "mrr": 0.9888268156424581,
  "recall_at_3": 0.8094972067039105,
  "pairwise_accuracy": 0.851685393258427,
  "invalid_action_rate": 0.1564245810055866,
  "unique_top1_actions": 3
}
```

Group bootstrap NDCG@3:

```json
{
  "unit": "student/group",
  "iterations": 1000,
  "seed": 2026,
  "mean": 0.9534039213618958,
  "ci_low_95": 0.9374497393578379,
  "ci_high_95": 0.9666358180908503
}
```

Action-level results are in `CROSSFITTED_ACTION_RESULTS.csv`. This evidence is not a new held-out result and does not inherit historical Panel B evidence.

## Integrity

- train/test group overlap: 0 in every outer fold
- 121 contaminated-design rows used: 0
- legacy H1 prediction: not used
- HPO/feature redesign/action redesign: none
- historical Panel B merged: no

## Claim boundary

Supported claim: the rebuilt ranker shows the reported cross-validated ranking behavior under group-safe nested/cross-fitted weak-label evaluation on the 179-row genuine Hybrid-OOF subset.

Not supported: final held-out recommendation performance, superiority over a held-out baseline, or safety-policy performance.
