# CNN-BiLSTM — OULAD — Final Results

All values come from frozen final outer-OOF or final probability-ensemble evidence. N/A means no frozen final prediction artifact exists; no screening metric or estimate is substituted.

Precision and Recall in the overall table are macro averages.

## Overall comparison

| Model | Accuracy | Balanced Accuracy | Precision | Recall | Macro-F1 | Weighted-F1 | Risk Precision | Risk Recall | Risk F1 | PR-AUC | ROC-AUC | Brier ↓ | NLL ↓ | ECE ↓ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| CNN-BiLSTM | 0.8401 | 0.8203 | 0.8431 | 0.8203 | 0.8281 | 0.8374 | 0.8522 | 0.7236 | 0.7826 | 0.8934 | N/A | 0.1134 | 0.3588 | 0.0087 |
| CNN-only | 0.8333 | 0.8124 | 0.8366 | 0.8124 | 0.8204 | 0.8302 | 0.8463 | 0.7100 | 0.7722 | 0.8884 | N/A | 0.1170 | 0.3706 | 0.0247 |
| BiLSTM-only | 0.8374 | 0.8223 | 0.8350 | 0.8223 | 0.8273 | 0.8358 | 0.8264 | 0.7484 | 0.7855 | 0.8904 | N/A | 0.1158 | 0.3688 | 0.0234 |
| Logistic Regression | N/A | 0.8214 | N/A | N/A | 0.8278 | N/A | 0.8390 | 0.7360 | 0.7842 | 0.8900 | N/A | 0.1148 | 0.3650 | 0.0077 |
| Decision Tree | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| Random Forest | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| HistGradientBoosting | N/A | 0.8207 | N/A | N/A | 0.8265 | N/A | 0.8323 | 0.7399 | 0.7834 | 0.8907 | N/A | 0.1163 | 0.3686 | 0.0333 |
| SVM | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| XGBoost | N/A | 0.8227 | N/A | N/A | 0.8284 | N/A | 0.8333 | 0.7437 | 0.7860 | 0.8922 | N/A | 0.1145 | 0.3624 | 0.0157 |

## Per-class results

| Model | Class | Precision | Recall | F1 | Support | Model Macro-F1 |
|---|---|---:|---:|---:|---:|---:|
| CNN-BiLSTM | Not-at-risk | 0.8339 | 0.9171 | 0.8735 | 9260 | 0.8281 |
| CNN-BiLSTM | At-risk | 0.8522 | 0.7236 | 0.7826 | 6118 | 0.8281 |
| CNN-only | Not-at-risk | 0.8268 | 0.9148 | 0.8686 | 9260 | 0.8204 |
| CNN-only | At-risk | 0.8463 | 0.7100 | 0.7722 | 6118 | 0.8204 |
| BiLSTM-only | Not-at-risk | 0.8435 | 0.8961 | 0.8690 | 9260 | 0.8273 |
| BiLSTM-only | At-risk | 0.8264 | 0.7484 | 0.7855 | 6118 | 0.8273 |
| Logistic Regression | Not-at-risk | N/A | N/A | N/A | N/A | N/A |
| Logistic Regression | At-risk | N/A | N/A | N/A | N/A | N/A |
| Decision Tree | Not-at-risk | N/A | N/A | N/A | N/A | N/A |
| Decision Tree | At-risk | N/A | N/A | N/A | N/A | N/A |
| Random Forest | Not-at-risk | N/A | N/A | N/A | N/A | N/A |
| Random Forest | At-risk | N/A | N/A | N/A | N/A | N/A |
| HistGradientBoosting | Not-at-risk | N/A | N/A | N/A | N/A | N/A |
| HistGradientBoosting | At-risk | N/A | N/A | N/A | N/A | N/A |
| SVM | Not-at-risk | N/A | N/A | N/A | N/A | N/A |
| SVM | At-risk | N/A | N/A | N/A | N/A | N/A |
| XGBoost | Not-at-risk | N/A | N/A | N/A | N/A | N/A |
| XGBoost | At-risk | N/A | N/A | N/A | N/A | N/A |

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
| Logistic Regression | 5% | N/A | N/A | N/A | N/A |
| Logistic Regression | 10% | N/A | N/A | N/A | N/A |
| Logistic Regression | 20% | N/A | N/A | N/A | N/A |
| Decision Tree | 5% | N/A | N/A | N/A | N/A |
| Decision Tree | 10% | N/A | N/A | N/A | N/A |
| Decision Tree | 20% | N/A | N/A | N/A | N/A |
| Random Forest | 5% | N/A | N/A | N/A | N/A |
| Random Forest | 10% | N/A | N/A | N/A | N/A |
| Random Forest | 20% | N/A | N/A | N/A | N/A |
| HistGradientBoosting | 5% | N/A | N/A | N/A | N/A |
| HistGradientBoosting | 10% | N/A | N/A | N/A | N/A |
| HistGradientBoosting | 20% | N/A | N/A | N/A | N/A |
| SVM | 5% | N/A | N/A | N/A | N/A |
| SVM | 10% | N/A | N/A | N/A | N/A |
| SVM | 20% | N/A | N/A | N/A | N/A |
| XGBoost | 5% | N/A | N/A | N/A | N/A |
| XGBoost | 10% | N/A | N/A | N/A | N/A |
| XGBoost | 20% | N/A | N/A | N/A | N/A |

## Frozen confusion matrices

### CNN-BiLSTM

```text
8492 768
1691 4427
```

## Evidence sources

- `artifacts/v5/oulad/final_metrics.csv` — SHA-256 `c666c2e741421359bc0ac329395a6200c5696045e5d1114c17258b6ea3965cb1`
- `artifacts/v5_1/oulad/final_metrics.json` — SHA-256 `c5c6c7615bbd7259b2445ce1055a815b58cd4a2a3124316353a788312fd729c1`
- `artifacts/v6/prediction/final/metrics.json` — SHA-256 `17110c49f9cde7d82d1583b051e9c13f2520062030e4954778b0196162c0067a`
