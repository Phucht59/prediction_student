# V19 Permutation Importance Report

Protocol: RandomForest proxy model on the strict split. Importance is measured by the drop in Macro-F1 after permuting each processed feature.

## student-mat

Baseline strict test Accuracy: 0.7833; Macro-F1: 0.7625.

| feature | importance_mean | importance_std |
| --- | --- | --- |
| numeric__G2 | 0.4827 | 0.0612 |
| numeric__G1 | 0.1019 | 0.0462 |

## student-por

Baseline strict test Accuracy: 0.6735; Macro-F1: 0.6541.

| feature | importance_mean | importance_std |
| --- | --- | --- |
| numeric__G2 | 0.3688 | 0.0563 |
| numeric__G1 | 0.1684 | 0.0595 |

## xapi

Baseline strict test Accuracy: 0.6944; Macro-F1: 0.7000.

| feature | importance_mean | importance_std |
| --- | --- | --- |
| numeric__StudentAbsenceDays | 0.2291 | 0.0683 |
| numeric__VisitedResources | 0.1701 | 0.0352 |
| numeric__raisedhands | 0.1241 | 0.0268 |

## Notes

- This is not SHAP and not a direct neural-network explanation; it is a reproducible permutation-importance proxy.
- It supports recommendation design by identifying which paper features most affect a baseline classifier.
