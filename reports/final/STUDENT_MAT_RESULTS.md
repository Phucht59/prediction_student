# CNN-BiLSTM — Student-Mat — Final Results

All values come from frozen final outer-OOF or final probability-ensemble evidence. N/A means no frozen final prediction artifact exists; no screening metric or estimate is substituted.

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
| XGBoost | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |

## Per-class results

| Model | Class | Precision | Recall | F1 | Support | Model Macro-F1 |
|---|---|---:|---:|---:|---:|---:|
| CNN-BiLSTM | Low | 0.8264 | 0.9154 | 0.8686 | 130 | 0.9015 |
| CNN-BiLSTM | Medium | 0.9116 | 0.8594 | 0.8847 | 192 | 0.9015 |
| CNN-BiLSTM | High | 0.9714 | 0.9315 | 0.9510 | 73 | 0.9015 |
| CNN-only | Low | 0.8444 | 0.8769 | 0.8604 | 130 | 0.8708 |
| CNN-only | Medium | 0.8798 | 0.8385 | 0.8587 | 192 | 0.8708 |
| CNN-only | High | 0.8701 | 0.9178 | 0.8933 | 73 | 0.8708 |
| BiLSTM-only | Low | 0.7919 | 0.9077 | 0.8459 | 130 | 0.8397 |
| BiLSTM-only | Medium | 0.8757 | 0.7708 | 0.8199 | 192 | 0.8397 |
| BiLSTM-only | High | 0.8312 | 0.8767 | 0.8533 | 73 | 0.8397 |
| Logistic Regression | Low | 0.8605 | 0.8538 | 0.8571 | 130 | 0.8793 |
| Logistic Regression | Medium | 0.8660 | 0.8750 | 0.8705 | 192 | 0.8793 |
| Logistic Regression | High | 0.9167 | 0.9041 | 0.9103 | 73 | 0.8793 |
| Decision Tree | Low | 0.8356 | 0.9385 | 0.8841 | 130 | 0.9067 |
| Decision Tree | Medium | 0.9176 | 0.8698 | 0.8930 | 192 | 0.9067 |
| Decision Tree | High | 0.9851 | 0.9041 | 0.9429 | 73 | 0.9067 |
| Random Forest | Low | 0.8345 | 0.9308 | 0.8800 | 130 | 0.9014 |
| Random Forest | Medium | 0.9167 | 0.8594 | 0.8871 | 192 | 0.9014 |
| Random Forest | High | 0.9571 | 0.9178 | 0.9371 | 73 | 0.9014 |
| HistGradientBoosting | Low | 0.8370 | 0.8692 | 0.8528 | 130 | 0.8785 |
| HistGradientBoosting | Medium | 0.8730 | 0.8594 | 0.8661 | 192 | 0.8785 |
| HistGradientBoosting | High | 0.9296 | 0.9041 | 0.9167 | 73 | 0.8785 |
| SVM | Low | 0.8160 | 0.7846 | 0.8000 | 130 | 0.8143 |
| SVM | Medium | 0.7970 | 0.8177 | 0.8072 | 192 | 0.8143 |
| SVM | High | 0.8356 | 0.8356 | 0.8356 | 73 | 0.8143 |
| XGBoost | Low | N/A | N/A | N/A | N/A | N/A |
| XGBoost | Medium | N/A | N/A | N/A | N/A | N/A |
| XGBoost | High | N/A | N/A | N/A | N/A | N/A |

## Frozen confusion matrices

### CNN-BiLSTM

```text
119 11 0
25 165 2
0 5 68
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

## Evidence sources

- `artifacts/v5_1/student_mat/final_metrics.json` — SHA-256 `ee4e998ca6e8173f374d78984ba5eba52105c1ae57501e93397bfc4e08aff776`
- `artifacts/v5_1/student_mat/ml_final_metrics.json` — SHA-256 `cb485ac2c3f8895ec628b4153ff2fbc0b76112b553032bbe0253527b0f0db726`
- `artifacts/v5_1/student_mat/ml_oof_predictions.parquet` — SHA-256 `7ba6855f87d83ef4e7f99e874de4e5d70e6e7164b70a3c5bf11a43d8877f8089`
- `artifacts/v5_1/student_mat/oof_predictions.parquet` — SHA-256 `1375a4f664a087b3c650624813bb2a01f5ff3586fc20b1688ca526c2c4dbf59e`
