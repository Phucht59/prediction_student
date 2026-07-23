# CNN-BiLSTM — Student-Por — Final Results

All values are recomputed from validated record-aligned outer-OOF probability ensembles. Frozen deep predictions are unchanged; explicitly identified comparators were trained under the preregistered completion protocol.

Precision and Recall in the overall table are macro averages.

## Overall comparison

| Model | Accuracy | Balanced Accuracy | Precision | Recall | Macro-F1 | Weighted-F1 | PR-AUC | ROC-AUC | Brier ↓ | NLL ↓ | ECE ↓ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| CNN-BiLSTM | 0.8891 | 0.8676 | 0.8573 | 0.8676 | 0.8623 | 0.8896 | 0.9147 | 0.9628 | 0.1725 | 0.3079 | 0.0537 |
| CNN-only | 0.8767 | 0.8518 | 0.8420 | 0.8518 | 0.8468 | 0.8773 | 0.9215 | 0.9636 | 0.1686 | 0.2959 | 0.0441 |
| BiLSTM-only | 0.8259 | 0.7986 | 0.7725 | 0.7986 | 0.7843 | 0.8262 | 0.8649 | 0.9418 | 0.2384 | 0.3952 | 0.0841 |
| Logistic Regression | 0.8521 | 0.8348 | 0.8081 | 0.8348 | 0.8205 | 0.8534 | 0.9125 | 0.9555 | 0.1945 | 0.3383 | 0.0385 |
| Decision Tree | 0.8706 | 0.8800 | 0.8283 | 0.8800 | 0.8487 | 0.8743 | 0.8966 | 0.9576 | 0.1777 | 0.3589 | 0.0319 |
| Random Forest | 0.8921 | 0.8836 | 0.8568 | 0.8836 | 0.8692 | 0.8933 | 0.9309 | 0.9689 | 0.1569 | 0.2722 | 0.0306 |
| HistGradientBoosting | 0.8829 | 0.8439 | 0.8578 | 0.8439 | 0.8506 | 0.8822 | 0.9023 | 0.9566 | 0.1790 | 0.3617 | 0.0553 |
| SVM | 0.8382 | 0.7612 | 0.8127 | 0.7612 | 0.7825 | 0.8332 | 0.8297 | 0.9103 | 0.2507 | 0.4872 | 0.0364 |
| XGBoost | 0.8952 | 0.8606 | 0.8731 | 0.8606 | 0.8664 | 0.8943 | 0.9361 | 0.9689 | 0.1512 | 0.2631 | 0.0281 |

## Per-class results

| Model | Class | Precision | Recall | F1 | Support |
|---|---|---:|---:|---:|---:|
| CNN-BiLSTM | Low | 0.7429 | 0.7800 | 0.7610 | 100 |
| CNN-BiLSTM | Medium | 0.9199 | 0.9067 | 0.9133 | 418 |
| CNN-BiLSTM | High | 0.9091 | 0.9160 | 0.9125 | 131 |
| CNN-only | Low | 0.7212 | 0.7500 | 0.7353 | 100 |
| CNN-only | Medium | 0.9102 | 0.8971 | 0.9036 | 418 |
| CNN-only | High | 0.8947 | 0.9084 | 0.9015 | 131 |
| BiLSTM-only | Low | 0.6176 | 0.6300 | 0.6238 | 100 |
| BiLSTM-only | Medium | 0.8822 | 0.8421 | 0.8617 | 418 |
| BiLSTM-only | High | 0.8176 | 0.9237 | 0.8674 | 131 |
| Logistic Regression | Low | 0.6789 | 0.7400 | 0.7081 | 100 |
| Logistic Regression | Medium | 0.9025 | 0.8636 | 0.8826 | 418 |
| Logistic Regression | High | 0.8429 | 0.9008 | 0.8708 | 131 |
| Decision Tree | Low | 0.6515 | 0.8600 | 0.7414 | 100 |
| Decision Tree | Medium | 0.9372 | 0.8565 | 0.8950 | 418 |
| Decision Tree | High | 0.8963 | 0.9237 | 0.9098 | 131 |
| Random Forest | Low | 0.7477 | 0.8300 | 0.7867 | 100 |
| Random Forest | Medium | 0.9328 | 0.8971 | 0.9146 | 418 |
| Random Forest | High | 0.8897 | 0.9237 | 0.9064 | 131 |
| HistGradientBoosting | Low | 0.7579 | 0.7200 | 0.7385 | 100 |
| HistGradientBoosting | Medium | 0.9014 | 0.9187 | 0.9100 | 418 |
| HistGradientBoosting | High | 0.9141 | 0.8931 | 0.9035 | 131 |
| SVM | Low | 0.7105 | 0.5400 | 0.6136 | 100 |
| SVM | Medium | 0.8486 | 0.9115 | 0.8789 | 418 |
| SVM | High | 0.8790 | 0.8321 | 0.8549 | 131 |
| XGBoost | Low | 0.8043 | 0.7400 | 0.7708 | 100 |
| XGBoost | Medium | 0.9127 | 0.9258 | 0.9192 | 418 |
| XGBoost | High | 0.9023 | 0.9160 | 0.9091 | 131 |

## Low-class analysis

CNN-BiLSTM Low precision/recall/F1 are 0.7429/0.7800/0.7610. Its frozen confusion matrix records 22 Low→Medium and 27 Medium→Low errors. Decision Tree and Random Forest Low-class results are shown in the same per-class table (0.7414 and 0.7867 F1 respectively).

## Confusion matrices

### CNN-BiLSTM

```text
78 22 0
27 379 12
0 11 120
```

### CNN-only

```text
75 25 0
29 375 14
0 12 119
```

### BiLSTM-only

```text
63 37 0
39 352 27
0 10 121
```

### Logistic Regression

```text
74 26 0
35 361 22
0 13 118
```

### Decision Tree

```text
86 14 0
46 358 14
0 10 121
```

### Random Forest

```text
83 17 0
28 375 15
0 10 121
```

### HistGradientBoosting

```text
72 28 0
23 384 11
0 14 117
```

### SVM

```text
54 46 0
22 381 15
0 22 109
```

### XGBoost

```text
74 26 0
18 387 13
0 11 120
```

## Evidence sources

- `artifacts/final/comparator_completion/student_por/oof_predictions.parquet` — SHA-256 `14258e818e14c9cf5b5bd21077453f85869bc5605984b17b03dfd4b818d72670`
