# Recommendation Validation

Recommendation is deterministic decision support, not Accuracy.

| Measure | Value |
|---|---:|
| Records | 15,378 |
| GENERATED | 10,953 |
| PARTIAL_EVIDENCE | 1,209 |
| ABSTAINED | 3,216 |
| Generated or partial | 79.09% |
| Abstention | 20.91% |

Workload violations, action-cap violations, duplicates, missing lineage,
post-cutoff usage, sensitive usage and withdrawal-mechanism usage are all zero.
Deterministic replay is PASS.

Every abstained case retains a traceable plan object with
`recommended_actions = []`.
