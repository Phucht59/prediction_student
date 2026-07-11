# Baselines and ablations

| Late-stage model | OOF Macro-F1 | Locked Macro-F1 |
| --- | ---: | ---: |
| Majority | 0.2184 | 0.2165 |
| G2 threshold rule | 0.8988 | 0.9365 |
| Logistic G2 | 0.8988 | 0.9365 |
| Logistic G1+G2 | 0.8664 | 0.8876 |
| Logistic all | 0.8197 | 0.8530 |
| HGB all | 0.8969 | 0.9463 |

| Deep variant | Params/model | OOF Macro-F1 | Locked Macro-F1 |
| --- | ---: | ---: | ---: |
| CNN-only | 115 | 0.8004 | 0.8582 |
| BiLSTM-only | 2,531 | 0.2184 | 0.2165 |
| CNN-BiLSTM fixed config | 4,515 | 0.8422 | 0.9098 |
| + class weight | 4,515 | 0.8007 | 0.8905 |
| + SMOTE | 4,515 | 0.8373 | 0.8797 |
| + SMOTE + class weight | 4,515 | 0.7634 | 0.8688 |
| 11-seed ensemble | 4,515 each | 0.8505 | 0.8876 |

The frozen selected model is a separately selected 13,059-parameter single-seed
configuration, not the fixed-config ablation row. The ensemble is not the final
deployable model. Early-warning best OOF Macro-F1 is 0.6974; pre-assessment best
OOF Macro-F1 is 0.4344.
# Scope correction

HGB full-feature and CNN–BiLSTM G1/G2 use different feature sets and tuning
levels. Treat their locked-test values as post-hoc system comparisons, not a
controlled architecture comparison. BiLSTM-only fixed ablation is not evidence
of causal CNN component importance.
