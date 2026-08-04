# Hybrid-only scientific postmortem

## Verified execution

Commit:

```text
bbc7e6905206feccfb8bb3744a7ad79a6034a547
```

Official status:

```text
HYBRID_ONLY_SILVER_EVIDENCE_BELOW_GATE
RECOMMENDATION_MODULE_NOT_COMPLETE
runtime_authorized = false
```

## Main results

```text
Ranking groups: 29,043
Candidate rows: 82,847
Groups with at least one positive future action: 9,304
Issued recommendations: 13,462
Correct issued recommendations: 3,650
Precision@1: 0.2711
Actionable coverage: 0.5186
Bootstrap Precision@1 95% CI: [0.2638, 0.2787]
```

The result is reproducible and not caused by one unstable fold:

```text
Fold 0 Precision@1: 0.2796
Fold 1 Precision@1: 0.2760
Fold 2 Precision@1: 0.2571
```

The inner selected configuration had Precision@1 `0.2709`, almost identical to held-out OOF `0.2711`. This indicates a structural feature/target limitation rather than threshold overfitting.

## Failure decomposition

Among 9,304 groups containing at least one positive future action, the system issued a recommendation for 4,825 groups. Among those issued-positive groups, 3,650 top actions were correct.

Therefore:

```text
Conditional top-action accuracy given an issued group with a positive action
= 3,650 / 4,825
= 0.7565
```

However, the system also issued recommendations for:

```text
13,462 - 4,825 = 8,637
```

groups in which no candidate action became positive in the observed future window.

The dominant failure is therefore not only action ranking. It is failure to distinguish:

```text
ACTIONABLE GROUP
vs
NO OBSERVED POSITIVE ACTION GROUP
```

## Evidence from the selected deterministic configuration

The selected configuration required:

```text
minimum_risk_reduction = 0.0
maximum_uncertainty = 1.0
minimum_top_score = 0.0
```

To preserve the 50% coverage gate, the selector could not rely on positive hybrid risk reduction, low uncertainty, or a positive absolute score. This is direct evidence that the current counterfactual risk features do not separate recommendable from non-recommendable groups.

## Stage failure

```text
EARLY_20 Precision@1: 0.1174
EARLY_35 Precision@1: 0.3110
MIDDLE_50 Precision@1: 0.3534
```

The earliest stage is especially weak. Sparse early behavior does not provide enough information for the current deterministic recommender to predict which action-specific behavior will improve in the next window.

## Comparison with existing supervised V2.1

The historical outcome-grounded V2.1 supervised ranker achieved approximately:

```text
Fold 0 model Precision@1: 0.6129
Fold 1 model Precision@1: 0.6140
Fold 2 model Precision@1: 0.5954
```

This confirms two points:

1. learned action scoring is materially stronger than the deterministic counterfactual formula;
2. a standard supervised ranker alone still does not support an 80% unconditional Precision@1 claim.

## Scientific conclusion

The frozen hybrid-only architecture is scientifically rejected for the registered 80% outcome-alignment objective.

Further weight tuning, larger threshold grids, or repeated runs are not justified because:

- inner and OOF performance agree;
- all outer folds fail similarly;
- scalar hybrid signals are weakly discriminative;
- the main error comes from issuing recommendations in groups with no observed positive future action.

## Required next architecture

The next defensible architecture is a two-stage selective recommender:

```text
Stage A — Recommendability / actionability gate
Predict whether at least one candidate action has a positive future label.

Stage B — Conditional action ranker
Only for groups passing Stage A, rank candidate actions and select top-1.
```

Evaluation must report separately:

```text
Stage A: precision, recall, PR-AUC, calibration and positive-group coverage
Stage B: conditional Precision@1, NDCG@3 and MRR on groups with a positive action
End-to-end: issued Precision@1, actionable coverage and abstention rate
```

An 80% claim is allowed only for the exact metric that reaches 80% on held-out folds. Conditional action accuracy must not be presented as unconditional end-to-end recommendation accuracy.

## Claim boundary

The release artifact currently records:

```text
HYBRID_MODEL_GUIDED_DECISION_SUPPORT_NOT_CAUSAL_EFFECT
```

The stronger general scientific boundary remains:

```text
OFFLINE_PREDICTIVE_RELEVANCE_NOT_CAUSAL_EFFECT
```

Neither wording permits causal-effect or guaranteed-grade-improvement claims.
