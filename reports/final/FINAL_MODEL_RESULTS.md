# Final Model Results

## Official models

| Dataset | Final model | Task |
|---|---|---|
| student-mat | CNN-BiLSTM — Student-Mat | multiclass_student_performance |
| student-por | CNN-BiLSTM — Student-Por | multiclass_student_performance |
| oulad | CNN-BiLSTM — OULAD | binary_student_risk |

## Student-Mat overall comparison

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

## Student-Mat per-class comparison

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

## Student-Por overall comparison

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
| XGBoost | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |

## Student-Por per-class comparison

| Model | Class | Precision | Recall | F1 | Support | Model Macro-F1 |
|---|---|---:|---:|---:|---:|---:|
| CNN-BiLSTM | Low | 0.7429 | 0.7800 | 0.7610 | 100 | 0.8623 |
| CNN-BiLSTM | Medium | 0.9199 | 0.9067 | 0.9133 | 418 | 0.8623 |
| CNN-BiLSTM | High | 0.9091 | 0.9160 | 0.9125 | 131 | 0.8623 |
| CNN-only | Low | 0.7212 | 0.7500 | 0.7353 | 100 | 0.8468 |
| CNN-only | Medium | 0.9102 | 0.8971 | 0.9036 | 418 | 0.8468 |
| CNN-only | High | 0.8947 | 0.9084 | 0.9015 | 131 | 0.8468 |
| BiLSTM-only | Low | 0.6176 | 0.6300 | 0.6238 | 100 | 0.7843 |
| BiLSTM-only | Medium | 0.8822 | 0.8421 | 0.8617 | 418 | 0.7843 |
| BiLSTM-only | High | 0.8176 | 0.9237 | 0.8674 | 131 | 0.7843 |
| Logistic Regression | Low | 0.6789 | 0.7400 | 0.7081 | 100 | 0.8205 |
| Logistic Regression | Medium | 0.9025 | 0.8636 | 0.8826 | 418 | 0.8205 |
| Logistic Regression | High | 0.8429 | 0.9008 | 0.8708 | 131 | 0.8205 |
| Decision Tree | Low | 0.6515 | 0.8600 | 0.7414 | 100 | 0.8487 |
| Decision Tree | Medium | 0.9372 | 0.8565 | 0.8950 | 418 | 0.8487 |
| Decision Tree | High | 0.8963 | 0.9237 | 0.9098 | 131 | 0.8487 |
| Random Forest | Low | 0.7477 | 0.8300 | 0.7867 | 100 | 0.8692 |
| Random Forest | Medium | 0.9328 | 0.8971 | 0.9146 | 418 | 0.8692 |
| Random Forest | High | 0.8897 | 0.9237 | 0.9064 | 131 | 0.8692 |
| HistGradientBoosting | Low | 0.7579 | 0.7200 | 0.7385 | 100 | 0.8506 |
| HistGradientBoosting | Medium | 0.9014 | 0.9187 | 0.9100 | 418 | 0.8506 |
| HistGradientBoosting | High | 0.9141 | 0.8931 | 0.9035 | 131 | 0.8506 |
| SVM | Low | 0.7105 | 0.5400 | 0.6136 | 100 | 0.7825 |
| SVM | Medium | 0.8486 | 0.9115 | 0.8789 | 418 | 0.7825 |
| SVM | High | 0.8790 | 0.8321 | 0.8549 | 131 | 0.7825 |
| XGBoost | Low | N/A | N/A | N/A | N/A | N/A |
| XGBoost | Medium | N/A | N/A | N/A | N/A | N/A |
| XGBoost | High | N/A | N/A | N/A | N/A | N/A |

## OULAD overall comparison

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

## OULAD per-class comparison

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

## OULAD Top-k

See `OULAD_RESULTS.md`; only rows backed by frozen probabilities contain values.

## Imbalance

See `IMBALANCE_RESULTS.md`.

## Recommendation

# Student Risk-Based Recommendation System — Final Results

This is a deterministic risk-based decision-support system, not a causal intervention claim.

| Measure | Value |
|---|---:|
| Plans Generated | 15378.0000 |
| Coverage | 1.0000 |
| Escalation Rate | 0.4686 |
| Conflicts | 0.0000 |
| Duplicate Plans | 0.0000 |
| Workload Violations | 0.0000 |
| Missing Lineage | 0.0000 |
| Deterministic Replay | True |
| Expert status | PENDING_EXPERT_LABELS |
| Causal effectiveness claimed | False |

Expert-label metrics remain N/A until independent labels are supplied.
