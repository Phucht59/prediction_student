# CNN-BiLSTM MAT — Final Results

All values are recomputed from validated record-aligned outer-OOF probability ensembles. Frozen deep predictions are unchanged; explicitly identified comparators were trained under the preregistered completion protocol.

Precision and Recall in the overall table are macro averages.

## Overall comparison

| Model | Accuracy | Balanced Accuracy | Precision | Recall | Macro-F1 | Weighted-F1 | PR-AUC | ROC-AUC | Brier ↓ | NLL ↓ | ECE ↓ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| CNN-BiLSTM | 0.8911 | 0.9021 | 0.9031 | 0.9021 | 0.9015 | 0.8917 | 0.9442 | 0.9679 | 0.2072 | 0.3635 | 0.1305 |
| CNN-only | 0.8658 | 0.8778 | 0.8648 | 0.8778 | 0.8708 | 0.8656 | 0.9300 | 0.9631 | 0.2278 | 0.3892 | 0.1222 |
| BiLSTM-only | 0.8354 | 0.8517 | 0.8330 | 0.8517 | 0.8397 | 0.8347 | 0.8950 | 0.9468 | 0.3069 | 0.5101 | 0.1808 |
| Logistic Regression | 0.8734 | 0.8777 | 0.8810 | 0.8777 | 0.8793 | 0.8735 | 0.9500 | 0.9699 | 0.1813 | 0.2952 | 0.0224 |
| Decision Tree | 0.8987 | 0.9041 | 0.9128 | 0.9041 | 0.9067 | 0.8993 | 0.8609 | 0.9300 | 0.1816 | 0.4272 | 0.0128 |
| Random Forest | 0.8937 | 0.9027 | 0.9028 | 0.9027 | 0.9014 | 0.8940 | 0.9550 | 0.9720 | 0.1669 | 0.2799 | 0.0328 |
| HistGradientBoosting | 0.8709 | 0.8776 | 0.8799 | 0.8776 | 0.8785 | 0.8711 | 0.9318 | 0.9609 | 0.1889 | 0.3593 | 0.0594 |
| SVM | 0.8101 | 0.8126 | 0.8162 | 0.8126 | 0.8143 | 0.8101 | 0.8827 | 0.9316 | 0.2702 | 0.4449 | 0.0491 |
| XGBoost | 0.8785 | 0.8828 | 0.8942 | 0.8828 | 0.8880 | 0.8790 | 0.9506 | 0.9689 | 0.1730 | 0.2961 | 0.0381 |

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
| Logistic Regression | Medium | 0.8660 | 0.8750 | 0.8705 | 192 |
| Logistic Regression | High | 0.9167 | 0.9041 | 0.9103 | 73 |
| Decision Tree | Low | 0.8356 | 0.9385 | 0.8841 | 130 |
| Decision Tree | Medium | 0.9176 | 0.8698 | 0.8930 | 192 |
| Decision Tree | High | 0.9851 | 0.9041 | 0.9429 | 73 |
| Random Forest | Low | 0.8345 | 0.9308 | 0.8800 | 130 |
| Random Forest | Medium | 0.9167 | 0.8594 | 0.8871 | 192 |
| Random Forest | High | 0.9571 | 0.9178 | 0.9371 | 73 |
| HistGradientBoosting | Low | 0.8370 | 0.8692 | 0.8528 | 130 |
| HistGradientBoosting | Medium | 0.8730 | 0.8594 | 0.8661 | 192 |
| HistGradientBoosting | High | 0.9296 | 0.9041 | 0.9167 | 73 |
| SVM | Low | 0.8160 | 0.7846 | 0.8000 | 130 |
| SVM | Medium | 0.7970 | 0.8177 | 0.8072 | 192 |
| SVM | High | 0.8356 | 0.8356 | 0.8356 | 73 |
| XGBoost | Low | 0.8370 | 0.8692 | 0.8528 | 130 |
| XGBoost | Medium | 0.8750 | 0.8750 | 0.8750 | 192 |
| XGBoost | High | 0.9706 | 0.9041 | 0.9362 | 73 |

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
18 168 6
0 7 66
```

### Decision Tree

```text
122 8 0
24 167 1
0 7 66
```

### Random Forest

```text
121 9 0
24 165 3
0 6 67
```

### HistGradientBoosting

```text
113 17 0
22 165 5
0 7 66
```

### SVM

```text
102 28 0
23 157 12
0 12 61
```

### XGBoost

```text
113 17 0
22 168 2
0 7 66
```

## Evidence sources

- `artifacts/final/comparator_completion/student_mat/oof_predictions.parquet` — SHA-256 `d7810e249a44d05230579db7362e49407874f0374b6f6e788978411ea7c8e76c`
