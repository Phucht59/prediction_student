# CNN-BiLSTM MAT — Final Results

All values are recomputed from validated record-aligned outer-OOF probability ensembles. Frozen deep predictions are unchanged; explicitly identified comparators were trained under the preregistered completion protocol.

Precision and Recall in the overall table are macro averages.

## Overall comparison

| Model | Accuracy | Balanced Accuracy | Precision | Recall | Macro-F1 | Weighted-F1 | PR-AUC | ROC-AUC | Brier ↓ | NLL ↓ | ECE ↓ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| CNN-BiLSTM | 0.8911 | 0.9021 | 0.9031 | 0.9021 | 0.9015 | 0.8917 | 0.9442 | 0.9679 | 0.2072 | 0.3635 | 0.1305 |
| CNN-only | 0.8658 | 0.8778 | 0.8648 | 0.8778 | 0.8708 | 0.8656 | 0.9300 | 0.9631 | 0.2278 | 0.3892 | 0.1222 |
| BiLSTM-only | 0.8354 | 0.8517 | 0.8330 | 0.8517 | 0.8397 | 0.8347 | 0.8950 | 0.9468 | 0.3069 | 0.5101 | 0.1808 |
| Logistic Regression | 0.8861 | 0.8977 | 0.8929 | 0.8977 | 0.8952 | 0.8859 | 0.9612 | 0.9736 | 0.1707 | 0.2728 | 0.0369 |
| Decision Tree | 0.8937 | 0.9018 | 0.9055 | 0.9018 | 0.9024 | 0.8942 | 0.9144 | 0.9539 | 0.1765 | 1.0421 | 0.0477 |
| Random Forest | 0.8911 | 0.8939 | 0.9073 | 0.8939 | 0.8998 | 0.8917 | 0.9625 | 0.9761 | 0.1594 | 0.2672 | 0.0321 |
| HistGradientBoosting | 0.8608 | 0.8673 | 0.8722 | 0.8673 | 0.8697 | 0.8609 | 0.9354 | 0.9593 | 0.2383 | 0.5281 | 0.1014 |
| SVM | 0.8633 | 0.8747 | 0.8676 | 0.8747 | 0.8710 | 0.8631 | 0.9508 | 0.9687 | 0.1855 | 0.3039 | 0.0300 |
| XGBoost | 0.8734 | 0.8785 | 0.8849 | 0.8785 | 0.8815 | 0.8737 | 0.9500 | 0.9711 | 0.1705 | 0.2814 | 0.0375 |
| MLP | 0.8532 | 0.8621 | 0.8570 | 0.8621 | 0.8595 | 0.8531 | 0.9503 | 0.9687 | 0.1985 | 0.3385 | 0.0797 |

## Per-class results

| Model | Class | Precision | Recall | F1 | Support |
|---|---|---:|---:|---:|---:|
| CNN-BiLSTM | Low | 0.8264 | 0.9154 | 0.8686 | 130 |
| CNN-BiLSTM | Medium | 0.9116 | 0.8594 | 0.8847 | 192 |
| CNN-BiLSTM | High | 0.9714 | 0.9315 | 0.9510 | 73 |
| CNN-only | Low | 0.8444 | 0.8769 | 0.8604 | 130 |
| CNN-only | Medium | 0.8798 | 0.8385 | 0.8587 | 192 |
| CNN-only | High | 0.8701 | 0.9178 | 0.8933 | 73 |
| BiLSTM-only | Low | 0.7919 | 0.9077 | 0.8459 | 130 |
| BiLSTM-only | Medium | 0.8757 | 0.7708 | 0.8199 | 192 |
| BiLSTM-only | High | 0.8312 | 0.8767 | 0.8533 | 73 |
| Logistic Regression | Low | 0.8605 | 0.8538 | 0.8571 | 130 |
| Logistic Regression | Medium | 0.8848 | 0.8802 | 0.8825 | 192 |
| Logistic Regression | High | 0.9333 | 0.9589 | 0.9459 | 73 |
| Decision Tree | Low | 0.8333 | 0.9231 | 0.8759 | 130 |
| Decision Tree | Medium | 0.9121 | 0.8646 | 0.8877 | 192 |
| Decision Tree | High | 0.9710 | 0.9178 | 0.9437 | 73 |
| Random Forest | Low | 0.8467 | 0.8923 | 0.8689 | 130 |
| Random Forest | Medium | 0.8901 | 0.8854 | 0.8877 | 192 |
| Random Forest | High | 0.9851 | 0.9041 | 0.9429 | 73 |
| HistGradientBoosting | Low | 0.8321 | 0.8385 | 0.8352 | 130 |
| HistGradientBoosting | Medium | 0.8549 | 0.8594 | 0.8571 | 192 |
| HistGradientBoosting | High | 0.9296 | 0.9041 | 0.9167 | 73 |
| SVM | Low | 0.8450 | 0.8385 | 0.8417 | 130 |
| SVM | Medium | 0.8632 | 0.8542 | 0.8586 | 192 |
| SVM | High | 0.8947 | 0.9315 | 0.9128 | 73 |
| XGBoost | Low | 0.8421 | 0.8615 | 0.8517 | 130 |
| XGBoost | Medium | 0.8698 | 0.8698 | 0.8698 | 192 |
| XGBoost | High | 0.9429 | 0.9041 | 0.9231 | 73 |
| MLP | Low | 0.8385 | 0.8385 | 0.8385 | 130 |
| MLP | Medium | 0.8526 | 0.8438 | 0.8482 | 192 |
| MLP | High | 0.8800 | 0.9041 | 0.8919 | 73 |

## Confusion matrices

### CNN-BiLSTM

```text
119 11 0
25 165 2
0 5 68
```

### CNN-only

```text
114 16 0
21 161 10
0 6 67
```

### BiLSTM-only

```text
118 12 0
31 148 13
0 9 64
```

### Logistic Regression

```text
111 19 0
18 169 5
0 3 70
```

### Decision Tree

```text
120 10 0
24 166 2
0 6 67
```

### Random Forest

```text
116 14 0
21 170 1
0 7 66
```

### HistGradientBoosting

```text
109 21 0
22 165 5
0 7 66
```

### SVM

```text
109 21 0
20 164 8
0 5 68
```

### XGBoost

```text
112 18 0
21 167 4
0 7 66
```

### MLP

```text
109 21 0
21 162 9
0 7 66
```

## Evidence sources

- `artifacts/final/comparator_completion/student_mat/oof_predictions.parquet` — SHA-256 `d7810e249a44d05230579db7362e49407874f0374b6f6e788978411ea7c8e76c`
- `artifacts/final/teacher_feedback_validation/safe_uci_comparators/student_mat/oof_predictions.parquet` — SHA-256 `a3741ec9cc8519c279369dc73f7b6eca949483767defbf83793c0ca3b20b828e`
