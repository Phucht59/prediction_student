# Final results

| Metric | Value |
| --- | ---: |
| Nested outer Macro-F1 | 0.8781 +/- 0.0448 |
| Locked accuracy / Macro-F1 / weighted F1 | 0.9114 / 0.9262 / 0.9122 |
| Balanced accuracy / QWK | 0.9345 / 0.9152 |
| Ordinal MAE; one-step; two-step errors | 0.0886; 7; 0 |
| Brier / ECE / macro PR-AUC | 0.1683 / 0.0591 / 0.9699 |

| Metric | Estimate | 95% CI |
| --- | ---: | ---: |
| Accuracy | 0.9114 | 0.8481–0.9620 |
| Macro-F1 | 0.9262 | 0.8704–0.9694 |

The intervals use 2,000 bootstrap resamples with seed 42. They reflect
sampling uncertainty from the 79-record locked test; the High class has only
15 records, so its observed perfect class score is not a general claim.

| Class | Precision | Recall | F1 | Support |
| --- | ---: | ---: | ---: | ---: |
| Low | 0.8065 | 0.9615 | 0.8772 | 26 |
| Medium | 0.9697 | 0.8421 | 0.9014 | 38 |
| High | 1.0000 | 1.0000 | 1.0000 | 15 |

Confusion matrix (true rows Low/Medium/High): `[[25,1,0],[6,32,0],[0,0,15]]`.
G2 beats final CNN-BiLSTM on locked Macro-F1 (0.9365 vs 0.9262). HGB reaches
0.9463 on locked test, but its 0.8969 OOF statistic is a different protocol;
the same-outer-fold HGB result is 0.8690. The correct conclusion is feasibility
and reproducibility, not demonstrated added value.
HGB full-feature and CNN–BiLSTM G1/G2 do not use the same feature set or the
same level of tuning, so this is not an absolutely controlled architecture comparison.
