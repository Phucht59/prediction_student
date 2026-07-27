# CNN-BiLSTM OULAD — Final Results

All values are recomputed from validated record-aligned outer-OOF probability ensembles. Frozen deep predictions are unchanged; explicitly identified comparators were trained under the preregistered completion protocol.

Precision and Recall in the overall table are macro averages.

## Overall comparison

| Model | Accuracy | Balanced Accuracy | Precision | Recall | Macro-F1 | Weighted-F1 | Risk Precision | Risk Recall | Risk F1 | PR-AUC | ROC-AUC | Brier ↓ | NLL ↓ | ECE ↓ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| CNN-BiLSTM | 0.8401 | 0.8203 | 0.8431 | 0.8203 | 0.8281 | 0.8374 | 0.8522 | 0.7236 | 0.7826 | 0.8934 | 0.9082 | 0.1134 | 0.3588 | 0.0087 |
| CNN-only | 0.8333 | 0.8124 | 0.8366 | 0.8124 | 0.8204 | 0.8302 | 0.8463 | 0.7100 | 0.7722 | 0.8884 | 0.9031 | 0.1170 | 0.3706 | 0.0247 |
| BiLSTM-only | 0.8374 | 0.8223 | 0.8350 | 0.8223 | 0.8273 | 0.8358 | 0.8264 | 0.7484 | 0.7855 | 0.8904 | 0.9053 | 0.1158 | 0.3688 | 0.0234 |
| Logistic Regression | 0.8353 | 0.8193 | 0.8336 | 0.8193 | 0.8247 | 0.8335 | 0.8274 | 0.7406 | 0.7816 | 0.8856 | 0.9028 | 0.1185 | 0.3759 | 0.0329 |
| Decision Tree | 0.8181 | 0.8007 | 0.8153 | 0.8007 | 0.8061 | 0.8160 | 0.8055 | 0.7156 | 0.7579 | 0.8587 | 0.8818 | 0.1301 | 0.4134 | 0.0203 |
| Random Forest | 0.8331 | 0.8161 | 0.8318 | 0.8161 | 0.8220 | 0.8311 | 0.8276 | 0.7331 | 0.7775 | 0.8847 | 0.8994 | 0.1188 | 0.3755 | 0.0179 |
| HistGradientBoosting | 0.8337 | 0.8203 | 0.8295 | 0.8203 | 0.8241 | 0.8325 | 0.8134 | 0.7551 | 0.7832 | 0.8889 | 0.9030 | 0.1173 | 0.3713 | 0.0313 |
| SVM | 0.8362 | 0.8186 | 0.8360 | 0.8186 | 0.8250 | 0.8340 | 0.8354 | 0.7326 | 0.7806 | 0.8797 | 0.8993 | 0.1184 | 0.3797 | 0.0171 |
| XGBoost | 0.8352 | 0.8222 | 0.8310 | 0.8222 | 0.8259 | 0.8341 | 0.8145 | 0.7586 | 0.7855 | 0.8900 | 0.9048 | 0.1159 | 0.3658 | 0.0160 |

## Per-class results

| Model | Class | Precision | Recall | F1 | Support |
|---|---|---:|---:|---:|---:|
| CNN-BiLSTM | Not-at-risk | 0.8339 | 0.9171 | 0.8735 | 9260 |
| CNN-BiLSTM | At-risk | 0.8522 | 0.7236 | 0.7826 | 6118 |
| CNN-only | Not-at-risk | 0.8268 | 0.9148 | 0.8686 | 9260 |
| CNN-only | At-risk | 0.8463 | 0.7100 | 0.7722 | 6118 |
| BiLSTM-only | Not-at-risk | 0.8435 | 0.8961 | 0.8690 | 9260 |
| BiLSTM-only | At-risk | 0.8264 | 0.7484 | 0.7855 | 6118 |
| Logistic Regression | Not-at-risk | 0.8397 | 0.8979 | 0.8679 | 9260 |
| Logistic Regression | At-risk | 0.8274 | 0.7406 | 0.7816 | 6118 |
| Decision Tree | Not-at-risk | 0.8250 | 0.8859 | 0.8543 | 9260 |
| Decision Tree | At-risk | 0.8055 | 0.7156 | 0.7579 | 6118 |
| Random Forest | Not-at-risk | 0.8360 | 0.8991 | 0.8664 | 9260 |
| Random Forest | At-risk | 0.8276 | 0.7331 | 0.7775 | 6118 |
| HistGradientBoosting | Not-at-risk | 0.8455 | 0.8855 | 0.8651 | 9260 |
| HistGradientBoosting | At-risk | 0.8134 | 0.7551 | 0.7832 | 6118 |
| SVM | Not-at-risk | 0.8366 | 0.9046 | 0.8693 | 9260 |
| SVM | At-risk | 0.8354 | 0.7326 | 0.7806 | 6118 |
| XGBoost | Not-at-risk | 0.8474 | 0.8859 | 0.8662 | 9260 |
| XGBoost | At-risk | 0.8145 | 0.7586 | 0.7855 | 6118 |

## Top-k risk ranking

Tie-breaking is descending probability then ascending record ID; the budget is rounded upward.

| Model | Budget | Precision@k | Recall@k | F1@k | NDCG@k |
|---|---:|---:|---:|---:|---:|
| CNN-BiLSTM | 5% | 0.9948 | 0.1250 | 0.2222 | 0.9953 |
| CNN-BiLSTM | 10% | 0.9961 | 0.2504 | 0.4002 | 0.9962 |
| CNN-BiLSTM | 20% | 0.9763 | 0.4908 | 0.6533 | 0.9791 |
| CNN-only | 5% | 0.9948 | 0.1250 | 0.2222 | 0.9955 |
| CNN-only | 10% | 0.9954 | 0.2502 | 0.3999 | 0.9957 |
| CNN-only | 20% | 0.9743 | 0.4899 | 0.6519 | 0.9774 |
| BiLSTM-only | 5% | 0.9948 | 0.1250 | 0.2222 | 0.9954 |
| BiLSTM-only | 10% | 0.9948 | 0.2501 | 0.3997 | 0.9951 |
| BiLSTM-only | 20% | 0.9733 | 0.4894 | 0.6513 | 0.9765 |
| Logistic Regression | 5% | 0.9948 | 0.1250 | 0.2222 | 0.9950 |
| Logistic Regression | 10% | 0.9909 | 0.2491 | 0.3981 | 0.9916 |
| Logistic Regression | 20% | 0.9665 | 0.4859 | 0.6467 | 0.9702 |
| Decision Tree | 5% | 0.9935 | 0.1249 | 0.2219 | 0.9934 |
| Decision Tree | 10% | 0.9831 | 0.2471 | 0.3950 | 0.9846 |
| Decision Tree | 20% | 0.9490 | 0.4771 | 0.6350 | 0.9545 |
| Random Forest | 5% | 0.9961 | 0.1252 | 0.2224 | 0.9963 |
| Random Forest | 10% | 0.9954 | 0.2502 | 0.3999 | 0.9957 |
| Random Forest | 20% | 0.9720 | 0.4887 | 0.6504 | 0.9754 |
| HistGradientBoosting | 5% | 0.9974 | 0.1254 | 0.2227 | 0.9976 |
| HistGradientBoosting | 10% | 0.9941 | 0.2499 | 0.3994 | 0.9947 |
| HistGradientBoosting | 20% | 0.9759 | 0.4907 | 0.6530 | 0.9787 |
| SVM | 5% | 0.9922 | 0.1247 | 0.2216 | 0.9931 |
| SVM | 10% | 0.9753 | 0.2452 | 0.3918 | 0.9783 |
| SVM | 20% | 0.9571 | 0.4812 | 0.6404 | 0.9611 |
| XGBoost | 5% | 0.9974 | 0.1254 | 0.2227 | 0.9979 |
| XGBoost | 10% | 0.9954 | 0.2502 | 0.3999 | 0.9960 |
| XGBoost | 20% | 0.9753 | 0.4904 | 0.6526 | 0.9783 |

## Confusion matrices

### CNN-BiLSTM

```text
8492 768
1691 4427
```

### CNN-only

```text
8471 789
1774 4344
```

### BiLSTM-only

```text
8298 962
1539 4579
```

### Logistic Regression

```text
8315 945
1587 4531
```

### Decision Tree

```text
8203 1057
1740 4378
```

### Random Forest

```text
8326 934
1633 4485
```

### HistGradientBoosting

```text
8200 1060
1498 4620
```

### SVM

```text
8377 883
1636 4482
```

### XGBoost

```text
8203 1057
1477 4641
```

## Evidence sources

- `artifacts/final/comparator_completion/oulad/ensemble_oof_predictions.parquet` — SHA-256 `c1e5dce42ad049db51caef8c6e7f0811e02236fb47db086a3d3e127299b1f041`
