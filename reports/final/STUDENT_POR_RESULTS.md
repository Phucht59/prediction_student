# CNN-BiLSTM POR — Final Results

All values are recomputed from validated record-aligned outer-OOF probability ensembles. Frozen deep predictions are unchanged; explicitly identified comparators were trained under the preregistered completion protocol.

Precision and Recall in the overall table are macro averages.

## Overall comparison

| Model | Accuracy | Balanced Accuracy | Precision | Recall | Macro-F1 | Weighted-F1 | PR-AUC | ROC-AUC | Brier ↓ | NLL ↓ | ECE ↓ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| CNN-BiLSTM | 0.8891 | 0.8676 | 0.8573 | 0.8676 | 0.8623 | 0.8896 | 0.9147 | 0.9628 | 0.1725 | 0.3079 | 0.0537 |
| CNN-only | 0.8767 | 0.8518 | 0.8420 | 0.8518 | 0.8468 | 0.8773 | 0.9215 | 0.9636 | 0.1686 | 0.2959 | 0.0441 |
| BiLSTM-only | 0.8259 | 0.7986 | 0.7725 | 0.7986 | 0.7843 | 0.8262 | 0.8649 | 0.9418 | 0.2384 | 0.3952 | 0.0841 |
| Logistic Regression | 0.8644 | 0.8581 | 0.8218 | 0.8581 | 0.8379 | 0.8666 | 0.9148 | 0.9550 | 0.1987 | 0.3374 | 0.0351 |
| Decision Tree | 0.8829 | 0.8295 | 0.8664 | 0.8295 | 0.8461 | 0.8807 | 0.8920 | 0.9432 | 0.1673 | 0.6404 | 0.0379 |
| Random Forest | 0.8798 | 0.8612 | 0.8424 | 0.8612 | 0.8514 | 0.8803 | 0.9291 | 0.9675 | 0.1609 | 0.2759 | 0.0321 |
| HistGradientBoosting | 0.8767 | 0.8433 | 0.8448 | 0.8433 | 0.8441 | 0.8767 | 0.8937 | 0.9533 | 0.1984 | 0.4732 | 0.0796 |
| SVM | 0.8829 | 0.8466 | 0.8543 | 0.8466 | 0.8502 | 0.8821 | 0.9192 | 0.9629 | 0.1695 | 0.2897 | 0.0237 |
| XGBoost | 0.8952 | 0.8657 | 0.8698 | 0.8657 | 0.8677 | 0.8949 | 0.9211 | 0.9605 | 0.1688 | 0.3067 | 0.0486 |
| MLP | 0.8706 | 0.8190 | 0.8461 | 0.8190 | 0.8304 | 0.8680 | 0.9147 | 0.9602 | 0.1735 | 0.3022 | 0.0475 |

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
| Logistic Regression | Low | 0.6780 | 0.8000 | 0.7339 | 100 |
| Logistic Regression | Medium | 0.9188 | 0.8660 | 0.8916 | 418 |
| Logistic Regression | High | 0.8686 | 0.9084 | 0.8881 | 131 |
| Decision Tree | Low | 0.7882 | 0.6700 | 0.7243 | 100 |
| Decision Tree | Medium | 0.8904 | 0.9330 | 0.9112 | 418 |
| Decision Tree | High | 0.9206 | 0.8855 | 0.9027 | 131 |
| Random Forest | Low | 0.7308 | 0.7600 | 0.7451 | 100 |
| Random Forest | Medium | 0.9187 | 0.8923 | 0.9053 | 418 |
| Random Forest | High | 0.8777 | 0.9313 | 0.9037 | 131 |
| HistGradientBoosting | Low | 0.7300 | 0.7300 | 0.7300 | 100 |
| HistGradientBoosting | Medium | 0.9045 | 0.9067 | 0.9056 | 418 |
| HistGradientBoosting | High | 0.9000 | 0.8931 | 0.8966 | 131 |
| SVM | Low | 0.7553 | 0.7100 | 0.7320 | 100 |
| SVM | Medium | 0.9052 | 0.9139 | 0.9095 | 418 |
| SVM | High | 0.9023 | 0.9160 | 0.9091 | 131 |
| XGBoost | Low | 0.7835 | 0.7600 | 0.7716 | 100 |
| XGBoost | Medium | 0.9167 | 0.9211 | 0.9189 | 418 |
| XGBoost | High | 0.9091 | 0.9160 | 0.9125 | 131 |
| MLP | Low | 0.7711 | 0.6400 | 0.6995 | 100 |
| MLP | Medium | 0.8866 | 0.9163 | 0.9012 | 418 |
| MLP | High | 0.8806 | 0.9008 | 0.8906 | 131 |

## Low-class analysis

CNN-BiLSTM Low precision/recall/F1 are 0.7429/0.7800/0.7610. Its frozen confusion matrix records 22 Low→Medium and 27 Medium→Low errors. Decision Tree and Random Forest Low-class results are shown in the same per-class table (0.7243 and 0.7451 F1 respectively).

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
80 20 0
38 362 18
0 12 119
```

### Decision Tree

```text
67 33 0
18 390 10
0 15 116
```

### Random Forest

```text
76 24 0
28 373 17
0 9 122
```

### HistGradientBoosting

```text
73 27 0
26 379 13
1 13 117
```

### SVM

```text
71 29 0
23 382 13
0 11 120
```

### XGBoost

```text
76 24 0
21 385 12
0 11 120
```

### MLP

```text
64 36 0
19 383 16
0 13 118
```

## Evidence sources

- `artifacts/final/comparator_completion/student_por/oof_predictions.parquet` — SHA-256 `14258e818e14c9cf5b5bd21077453f85869bc5605984b17b03dfd4b818d72670`
- `artifacts/final/teacher_feedback_validation/safe_uci_comparators/student_por/oof_predictions.parquet` — SHA-256 `1cc57e92c2c5625d3728bae6db0fd9b778356ba80d2c7f64e38c30b85120877e`
