# Final Model Results

## Official models

| Dataset | Final model | Task |
|---|---|---|
| student-mat | CNN-BiLSTM MAT | multiclass_student_performance |
| student-por | CNN-BiLSTM POR | multiclass_student_performance |
| oulad | CNN-BiLSTM OULAD | binary_student_risk |

## Student-Mat overall comparison

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

## Student-Mat per-class comparison

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

## Student-Por overall comparison

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

## Student-Por per-class comparison

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

## OULAD overall comparison

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
| MLP | 0.8393 | 0.8219 | 0.8392 | 0.8219 | 0.8283 | 0.8372 | 0.8392 | 0.7372 | 0.7849 | 0.8917 | 0.9073 | 0.2287 | 0.3620 | 0.0060 |

## OULAD per-class comparison

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
| MLP | Not-at-risk | 0.8393 | 0.9067 | 0.8717 | 9260 |
| MLP | At-risk | 0.8392 | 0.7372 | 0.7849 | 6118 |

## OULAD Top-k

See `OULAD_RESULTS.md`; all ten models have aligned probability-based budget results.

## Statistical comparison

See `COMPARATOR_COMPLETION_REPORT.md` and `MLP_COMPARATOR_REPORT.md` for paired bootstrap intervals on all three datasets.

## Imbalance

See `IMBALANCE_RESULTS.md`.

## Recommendation

# Student Risk-Based Recommendation System — Final Results

This is a deterministic risk-based decision-support system, not a causal intervention claim.

| Measure | Value |
|---|---:|
| Records | 15378.0000 |
| Generated | 10953.0000 |
| Partial Evidence | 1209.0000 |
| Abstained | 3216.0000 |
| Generated Or Partial | 0.7909 |
| Abstention | 0.2091 |
| Workload Violations | 0.0000 |
| Action Cap Violations | 0.0000 |
| Duplicates | 0.0000 |
| Missing Lineage | 0.0000 |
| Post Cutoff Usage | False |
| Sensitive Usage | False |
| Withdrawal Mechanism Usage | 0.0000 |
| Deterministic Replay | True |
| Expert status | PENDING_EXPERT_LABELS |
| Causal effectiveness claimed | False |

Expert-label metrics remain N/A until independent labels are supplied.
