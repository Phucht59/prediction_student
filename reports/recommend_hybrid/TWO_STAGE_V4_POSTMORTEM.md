# Two-Stage V4 scientific postmortem

## Registered outcome

```text
Execution commit: 5550f590e5afa69af54dd2c6d05ea2c9c22aabb6
Status: TWO_STAGE_V4_EVIDENCE_BELOW_GATE
Runtime authorized: false
```

Held-out OOF evidence:

```text
End-to-end Precision@1: 0.6589
Learner-cluster 95% CI: [0.6473, 0.6701]
Positive-group coverage: 0.4980
Stage A precision: 0.6953
Stage B conditional Precision@1: 0.9476
Ranking-only Precision@1: 0.9374
NDCG@3: 0.9723
MRR: 0.9669
```

The release gate remains failed. Conditional action-ranking performance may not be reported as end-to-end recommendation precision.

## What V4 changed

V4 corrected the registered V3 loss defect by applying candidate binary supervision to every valid candidate, including all-zero targets in negative groups. It also introduced action-derived recommendability through masked noisy-OR and a preregistered direct/action probability blend.

No labels, candidates, outer folds, frozen CNN–BiLSTM checkpoints, protected-feature rules, or release gates changed.

## What the evidence shows

V4 improved end-to-end Precision@1 from 0.6431 to 0.6589, an absolute gain of 0.0158. The improvement is stable but far below the required 0.80.

The direct recommendability head remained the strongest gate signal:

```text
Direct gate ROC-AUC / AP: 0.8109 / 0.6680
Action-derived ROC-AUC / AP: 0.8070 / 0.6471
Joint gate ROC-AUC / AP: 0.8084 / 0.6527
```

The action-derived signal did not add discrimination beyond the direct gate. Fold 2 selected a direct-only blend, while folds 0 and 1 used only a 0.25 direct weight.

With conditional Precision@1 = 0.9476, Stage A precision would need approximately:

```text
0.80 / 0.9476 = 0.8443
```

The observed Stage A precision was 0.6953, with bootstrap upper 95% bound approximately 0.7060. This is not a threshold-sized gap.

## Stable failure pattern

All outer folds remain in a narrow band:

```text
Fold 0 end-to-end Precision@1: 0.6565
Fold 1 end-to-end Precision@1: 0.6628
Fold 2 end-to-end Precision@1: 0.6578
```

This consistency indicates a representation/target boundary rather than a random split failure.

Stage B is no longer the limiting component:

```text
Conditional Precision@1: 0.9476
Ranking-only Precision@1: 0.9374
NDCG@3: 0.9723
```

The dominant limitation remains false-issue control: the system cannot identify groups with at least one observed positive future action at the precision needed by the original end-to-end gate while preserving 50% positive-group coverage.

## Action-level interpretation

`ASSESSMENT_COMPLETION` remained visible but was never issued. The other actions achieved high conditional precision but substantially lower issued-action precision because false issue groups dominate the error:

```text
STUDY_REGULARITY conditional P@1: 0.9100; issued-action precision: 0.5497
VLE_ENGAGEMENT conditional P@1: 0.9656; issued-action precision: 0.6974
QUIZ_OR_RETRIEVAL_PRACTICE conditional P@1: 0.8889; issued-action precision: 0.6220
CONTENT_REVIEW conditional P@1: 0.9536; issued-action precision: 0.6813
```

## Next scientific decision

No V4.1 threshold or loss-weight search should be started until the post-hoc feasibility audit is complete. The audit may estimate whether the current OOF score family has any precision–coverage point capable of reaching the original gate, but it is diagnostic and cannot authorize release.

If the optimistic registered-grid oracle remains below 0.80 at coverage 0.50, the current frozen-representation end-to-end problem is exhausted. The remaining defensible choices are:

1. introduce a genuinely new cutoff-safe representation and preregister a new experiment; or
2. define the recommendation module as conditional action ranking behind an external hybrid-risk/policy eligibility boundary, while preserving the failed end-to-end experiment as immutable evidence.

The second choice would be a different module boundary, not a claim that V4 passed its original gate.

## Claim boundary

```text
OFFLINE_PREDICTIVE_RELEVANCE_NOT_CAUSAL_EFFECT
```

No causal-effect, guaranteed-grade-improvement, expert-validation, production-readiness, or autonomous-runtime claim is permitted.
