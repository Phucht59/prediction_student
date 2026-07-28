# UCI Timing Scenario Report

## 1. Target definition

The target is derived only from final grade G3: Low 0–9, Medium 10–14, and High 15–20. G1/G2 are time-available predictors, never target labels.

## 2. Timing definition

- S0: before G1, 12 context fields only.
- S1: after G1 and before G2, context plus G1.
- S2: after G2 and before G3, context plus G1/G2; this is late-stage prediction.

## 3. G1/G2 band relationship with G3

### student_mat

G1→G2 same/improved/declined: 0.792 / 0.101 / 0.106. G2→G3 same-band rate: 0.899.

G1 band × G3 target counts: `[[104, 38, 0], [26, 143, 19], [0, 11, 54]]`

G2 band × G3 target counts: `[[122, 24, 0], [8, 167, 7], [0, 1, 66]]`

### student_por

G1→G2 same/improved/declined: 0.792 / 0.126 / 0.082. G2→G3 same-band rate: 0.843.

G1 band × G3 target counts: `[[88, 69, 0], [12, 343, 56], [0, 6, 75]]`

G2 band × G3 target counts: `[[89, 56, 0], [11, 361, 34], [0, 1, 97]]`

## 4. Fair comparison protocol

All ten models use identical frozen outer rows and information sources within each dataset/scenario. Preprocessing is fit on training rows only; selection uses three inner folds; outer rows are never used for tuning. The fair deep models use no transfer or pretrained UCI checkpoint.

## 5. student_mat — S0_EARLY_NO_GRADE

| Model | Macro-F1 | Bal Acc | Low P | Low R | Low F1 | PR-AUC | ECE | Seed SD |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Random Forest | 0.4595 | 0.4651 | 0.5118 | 0.5000 | 0.5058 | 0.4513 | 0.0772 | 0.0087 |
| XGBoost | 0.4436 | 0.4361 | 0.4911 | 0.4231 | 0.4545 | 0.4529 | 0.1913 | 0.0059 |
| HistGradientBoosting | 0.4267 | 0.4221 | 0.4419 | 0.4385 | 0.4402 | 0.4413 | 0.2771 | 0.0000 |
| CNN-BiLSTM | 0.4157 | 0.4376 | 0.5294 | 0.4154 | 0.4655 | 0.4465 | 0.0502 | 0.0121 |
| Decision Tree | 0.4154 | 0.4150 | 0.4701 | 0.4846 | 0.4773 | 0.4193 | 0.2173 | 0.0017 |
| Logistic Regression | 0.4053 | 0.4326 | 0.5000 | 0.4231 | 0.4583 | 0.4501 | 0.1121 | 0.0000 |
| CNN-only | 0.4037 | 0.4361 | 0.5405 | 0.4615 | 0.4979 | 0.4329 | 0.0306 | 0.0121 |
| MLP | 0.4022 | 0.4203 | 0.5750 | 0.3538 | 0.4381 | 0.4564 | 0.0799 | 0.0253 |
| SVM | 0.3751 | 0.4090 | 0.6301 | 0.3538 | 0.4532 | 0.4553 | 0.0535 | 0.0026 |
| BiLSTM-only | 0.3710 | 0.4405 | 0.5109 | 0.5385 | 0.5243 | 0.4377 | 0.0534 | 0.0434 |

Grade-band reference: Macro-F1 0.2181, Low Recall 0.0000, Low F1 0.0000.

## 6. student_mat — S1_MID_G1_ONLY

| Model | Macro-F1 | Bal Acc | Low P | Low R | Low F1 | PR-AUC | ECE | Seed SD |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| SVM | 0.7516 | 0.7501 | 0.7442 | 0.7385 | 0.7413 | 0.8367 | 0.0522 | 0.0028 |
| MLP | 0.7466 | 0.7393 | 0.7344 | 0.7231 | 0.7287 | 0.8252 | 0.0727 | 0.0112 |
| Logistic Regression | 0.7444 | 0.7441 | 0.7364 | 0.7308 | 0.7336 | 0.7976 | 0.0529 | 0.0000 |
| Decision Tree | 0.7426 | 0.7365 | 0.7581 | 0.7231 | 0.7402 | 0.7956 | 0.0696 | 0.0014 |
| Random Forest | 0.7233 | 0.7241 | 0.6849 | 0.7692 | 0.7246 | 0.8218 | 0.0514 | 0.0019 |
| XGBoost | 0.7212 | 0.7144 | 0.6846 | 0.6846 | 0.6846 | 0.8151 | 0.0885 | 0.0024 |
| CNN-BiLSTM | 0.7176 | 0.7658 | 0.6686 | 0.9000 | 0.7672 | 0.8243 | 0.0503 | 0.0225 |
| CNN-only | 0.7128 | 0.7694 | 0.6704 | 0.9231 | 0.7767 | 0.8140 | 0.0515 | 0.0165 |
| HistGradientBoosting | 0.6900 | 0.6802 | 0.7143 | 0.6538 | 0.6827 | 0.7698 | 0.1773 | 0.0000 |
| BiLSTM-only | 0.6588 | 0.7387 | 0.6448 | 0.9077 | 0.7540 | 0.7775 | 0.0526 | 0.0154 |

Grade-band reference: Macro-F1 0.7666, Low Recall 0.8000, Low F1 0.7647.

## 7. student_mat — S2_LATE_G1_G2

| Model | Macro-F1 | Bal Acc | Low P | Low R | Low F1 | PR-AUC | ECE | Seed SD |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Decision Tree | 0.9024 | 0.9018 | 0.8333 | 0.9231 | 0.8759 | 0.9144 | 0.0477 | 0.0000 |
| Random Forest | 0.8998 | 0.8939 | 0.8467 | 0.8923 | 0.8689 | 0.9625 | 0.0321 | 0.0020 |
| Logistic Regression | 0.8952 | 0.8977 | 0.8605 | 0.8538 | 0.8571 | 0.9612 | 0.0369 | 0.0000 |
| CNN-BiLSTM | 0.8829 | 0.8973 | 0.8264 | 0.9154 | 0.8686 | 0.9529 | 0.0555 | 0.0087 |
| XGBoost | 0.8815 | 0.8785 | 0.8421 | 0.8615 | 0.8517 | 0.9500 | 0.0375 | 0.0017 |
| SVM | 0.8710 | 0.8747 | 0.8450 | 0.8385 | 0.8417 | 0.9508 | 0.0300 | 0.0020 |
| HistGradientBoosting | 0.8697 | 0.8673 | 0.8321 | 0.8385 | 0.8352 | 0.9354 | 0.1014 | 0.0000 |
| MLP | 0.8595 | 0.8621 | 0.8385 | 0.8385 | 0.8385 | 0.9503 | 0.0797 | 0.0078 |
| CNN-only | 0.8526 | 0.8771 | 0.8013 | 0.9308 | 0.8612 | 0.9460 | 0.0802 | 0.0083 |
| BiLSTM-only | 0.8453 | 0.8727 | 0.7871 | 0.9385 | 0.8561 | 0.9420 | 0.0759 | 0.0129 |

Grade-band reference: Macro-F1 0.9067, Low Recall 0.9385, Low F1 0.8841.

## 8. student_por — S0_EARLY_NO_GRADE

| Model | Macro-F1 | Bal Acc | Low P | Low R | Low F1 | PR-AUC | ECE | Seed SD |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Random Forest | 0.5180 | 0.5641 | 0.4297 | 0.5500 | 0.4825 | 0.5117 | 0.0447 | 0.0065 |
| Logistic Regression | 0.5000 | 0.5878 | 0.4632 | 0.6300 | 0.5339 | 0.5110 | 0.1133 | 0.0000 |
| BiLSTM-only | 0.4689 | 0.5721 | 0.4444 | 0.6400 | 0.5246 | 0.4819 | 0.0912 | 0.0255 |
| Decision Tree | 0.4612 | 0.4970 | 0.3358 | 0.4600 | 0.3882 | 0.4394 | 0.2541 | 0.0026 |
| CNN-BiLSTM | 0.4571 | 0.5675 | 0.4238 | 0.6400 | 0.5100 | 0.4815 | 0.0988 | 0.0181 |
| HistGradientBoosting | 0.4515 | 0.4393 | 0.3714 | 0.2600 | 0.3059 | 0.4613 | 0.2126 | 0.0000 |
| CNN-only | 0.4502 | 0.5459 | 0.3961 | 0.6100 | 0.4803 | 0.4781 | 0.0893 | 0.0512 |
| XGBoost | 0.4498 | 0.4351 | 0.4035 | 0.2300 | 0.2930 | 0.4851 | 0.1204 | 0.0024 |
| MLP | 0.3433 | 0.3693 | 0.5000 | 0.1400 | 0.2188 | 0.4841 | 0.0277 | 0.0161 |
| SVM | 0.3161 | 0.3541 | 0.3684 | 0.0700 | 0.1176 | 0.4845 | 0.0565 | 0.0057 |

Grade-band reference: Macro-F1 0.2612, Low Recall 0.0000, Low F1 0.0000.

## 9. student_por — S1_MID_G1_ONLY

| Model | Macro-F1 | Bal Acc | Low P | Low R | Low F1 | PR-AUC | ECE | Seed SD |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Decision Tree | 0.8101 | 0.8137 | 0.7609 | 0.7000 | 0.7292 | 0.7741 | 0.0310 | 0.0000 |
| Random Forest | 0.7949 | 0.8123 | 0.6909 | 0.7600 | 0.7238 | 0.8183 | 0.0499 | 0.0031 |
| XGBoost | 0.7771 | 0.7723 | 0.7158 | 0.6800 | 0.6974 | 0.8239 | 0.0418 | 0.0017 |
| Logistic Regression | 0.7688 | 0.7945 | 0.6271 | 0.7400 | 0.6789 | 0.8286 | 0.0369 | 0.0000 |
| CNN-BiLSTM | 0.7539 | 0.8201 | 0.5577 | 0.8700 | 0.6797 | 0.8373 | 0.0468 | 0.0093 |
| SVM | 0.7525 | 0.7414 | 0.6778 | 0.6100 | 0.6421 | 0.8112 | 0.0508 | 0.0031 |
| MLP | 0.7440 | 0.7234 | 0.6786 | 0.5700 | 0.6196 | 0.8280 | 0.0333 | 0.0159 |
| CNN-only | 0.7384 | 0.8166 | 0.5500 | 0.8800 | 0.6769 | 0.8294 | 0.0489 | 0.0108 |
| HistGradientBoosting | 0.7320 | 0.7186 | 0.6630 | 0.6100 | 0.6354 | 0.7655 | 0.1314 | 0.0000 |
| BiLSTM-only | 0.6977 | 0.7961 | 0.5087 | 0.8800 | 0.6447 | 0.7911 | 0.0666 | 0.0177 |

Grade-band reference: Macro-F1 0.7400, Low Recall 0.8800, Low F1 0.6848.

## 10. student_por — S2_LATE_G1_G2

| Model | Macro-F1 | Bal Acc | Low P | Low R | Low F1 | PR-AUC | ECE | Seed SD |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| XGBoost | 0.8677 | 0.8657 | 0.7835 | 0.7600 | 0.7716 | 0.9211 | 0.0486 | 0.0025 |
| Random Forest | 0.8514 | 0.8612 | 0.7308 | 0.7600 | 0.7451 | 0.9291 | 0.0321 | 0.0013 |
| SVM | 0.8502 | 0.8466 | 0.7553 | 0.7100 | 0.7320 | 0.9192 | 0.0237 | 0.0003 |
| Decision Tree | 0.8461 | 0.8295 | 0.7882 | 0.6700 | 0.7243 | 0.8920 | 0.0379 | 0.0000 |
| CNN-only | 0.8460 | 0.8915 | 0.6718 | 0.8800 | 0.7619 | 0.9190 | 0.0236 | 0.0107 |
| CNN-BiLSTM | 0.8446 | 0.8957 | 0.6364 | 0.9100 | 0.7490 | 0.9227 | 0.0287 | 0.0039 |
| HistGradientBoosting | 0.8441 | 0.8433 | 0.7300 | 0.7300 | 0.7300 | 0.8937 | 0.0796 | 0.0000 |
| Logistic Regression | 0.8379 | 0.8581 | 0.6780 | 0.8000 | 0.7339 | 0.9148 | 0.0351 | 0.0000 |
| BiLSTM-only | 0.8356 | 0.8927 | 0.6233 | 0.9100 | 0.7398 | 0.9054 | 0.0337 | 0.0154 |
| MLP | 0.8304 | 0.8190 | 0.7711 | 0.6400 | 0.6995 | 0.9147 | 0.0475 | 0.0058 |

Grade-band reference: Macro-F1 0.8166, Low Recall 0.8900, Low F1 0.7265.

## 11. Early-warning answer

- student_mat S0_EARLY_NO_GRADE: best Macro-F1 Random Forest (0.4595); best Low Recall BiLSTM-only (0.5385); best Low F1 BiLSTM-only (0.5243).
- student_mat S1_MID_G1_ONLY: best Macro-F1 SVM (0.7516); best Low Recall CNN-only (0.9231); best Low F1 CNN-only (0.7767).
- student_por S0_EARLY_NO_GRADE: best Macro-F1 Random Forest (0.5180); best Low Recall CNN-BiLSTM (0.6400); best Low F1 Logistic Regression (0.5339).
- student_por S1_MID_G1_ONLY: best Macro-F1 Decision Tree (0.8101); best Low Recall CNN-only (0.8800); best Low F1 Decision Tree (0.7292).

## 12. Hybrid strength profile

CNN-BiLSTM strengths are reported only where the paired 5,000-replicate confidence interval excludes zero. Full results are in `paired_bootstrap_all_models.csv`; intervals crossing zero mean insufficient evidence of a difference, not equivalence.

- student_mat S0_EARLY_NO_GRADE low_f1: 1 comparisons — CNN-BiLSTM higher.
- student_mat S0_EARLY_NO_GRADE low_f1: 9 comparisons — insufficient evidence of difference.
- student_mat S0_EARLY_NO_GRADE low_recall: 1 comparisons — CNN-BiLSTM higher.
- student_mat S0_EARLY_NO_GRADE low_recall: 1 comparisons — comparator higher.
- student_mat S0_EARLY_NO_GRADE low_recall: 8 comparisons — insufficient evidence of difference.
- student_mat S0_EARLY_NO_GRADE macro_f1: 1 comparisons — CNN-BiLSTM higher.
- student_mat S0_EARLY_NO_GRADE macro_f1: 9 comparisons — insufficient evidence of difference.
- student_mat S1_MID_G1_ONLY low_f1: 3 comparisons — CNN-BiLSTM higher.
- student_mat S1_MID_G1_ONLY low_f1: 7 comparisons — insufficient evidence of difference.
- student_mat S1_MID_G1_ONLY low_recall: 8 comparisons — CNN-BiLSTM higher.
- student_mat S1_MID_G1_ONLY low_recall: 2 comparisons — insufficient evidence of difference.
- student_mat S1_MID_G1_ONLY macro_f1: 1 comparisons — CNN-BiLSTM higher.
- student_mat S1_MID_G1_ONLY macro_f1: 1 comparisons — comparator higher.
- student_mat S1_MID_G1_ONLY macro_f1: 8 comparisons — insufficient evidence of difference.
- student_mat S2_LATE_G1_G2 low_f1: 1 comparisons — CNN-BiLSTM higher.
- student_mat S2_LATE_G1_G2 low_f1: 1 comparisons — comparator higher.
- student_mat S2_LATE_G1_G2 low_f1: 8 comparisons — insufficient evidence of difference.
- student_mat S2_LATE_G1_G2 low_recall: 5 comparisons — CNN-BiLSTM higher.
- student_mat S2_LATE_G1_G2 low_recall: 5 comparisons — insufficient evidence of difference.
- student_mat S2_LATE_G1_G2 macro_f1: 2 comparisons — CNN-BiLSTM higher.
- student_mat S2_LATE_G1_G2 macro_f1: 1 comparisons — comparator higher.
- student_mat S2_LATE_G1_G2 macro_f1: 7 comparisons — insufficient evidence of difference.
- student_por S0_EARLY_NO_GRADE low_f1: 6 comparisons — CNN-BiLSTM higher.
- student_por S0_EARLY_NO_GRADE low_f1: 1 comparisons — comparator higher.
- student_por S0_EARLY_NO_GRADE low_f1: 3 comparisons — insufficient evidence of difference.
- student_por S0_EARLY_NO_GRADE low_recall: 7 comparisons — CNN-BiLSTM higher.
- student_por S0_EARLY_NO_GRADE low_recall: 3 comparisons — insufficient evidence of difference.
- student_por S0_EARLY_NO_GRADE macro_f1: 3 comparisons — CNN-BiLSTM higher.
- student_por S0_EARLY_NO_GRADE macro_f1: 2 comparisons — comparator higher.
- student_por S0_EARLY_NO_GRADE macro_f1: 5 comparisons — insufficient evidence of difference.
- student_por S1_MID_G1_ONLY low_f1: 10 comparisons — insufficient evidence of difference.
- student_por S1_MID_G1_ONLY low_recall: 7 comparisons — CNN-BiLSTM higher.
- student_por S1_MID_G1_ONLY low_recall: 3 comparisons — insufficient evidence of difference.
- student_por S1_MID_G1_ONLY macro_f1: 2 comparisons — CNN-BiLSTM higher.
- student_por S1_MID_G1_ONLY macro_f1: 2 comparisons — comparator higher.
- student_por S1_MID_G1_ONLY macro_f1: 6 comparisons — insufficient evidence of difference.
- student_por S2_LATE_G1_G2 low_f1: 10 comparisons — insufficient evidence of difference.
- student_por S2_LATE_G1_G2 low_recall: 7 comparisons — CNN-BiLSTM higher.
- student_por S2_LATE_G1_G2 low_recall: 3 comparisons — insufficient evidence of difference.
- student_por S2_LATE_G1_G2 macro_f1: 10 comparisons — insufficient evidence of difference.

## 13. Claim boundaries

- S0/S1 support early-warning claims; S2 is late-stage.
- Low is the final G3 class, not a timing label.
- Fair timing deep models use no Student-Por→Student-Mat transfer.
- Official frozen models answer a different final-model question and are unchanged.
- No result establishes universal CNN-BiLSTM superiority.
