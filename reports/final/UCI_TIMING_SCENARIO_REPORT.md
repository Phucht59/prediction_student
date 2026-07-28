# UCI Timing Scenario Report

This is a diagnostic information-timing study. It does not reselect or retrain either official UCI CNN-BiLSTM model.

S0 uses context only, S1 adds G1, and S2 adds G2 using the frozen two-timestep feature contract. G3 is target-only.

## Student-Mat

| Model | Scenario | Accuracy | Balanced Accuracy | Macro-F1 | Low Recall | PR-AUC | ECE |
|---|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | S0_EARLY_NO_GRADE | 0.4076 | 0.4326 | 0.4053 | 0.4231 | 0.4501 | 0.1121 |
| Logistic Regression | S1_MID_G1_ONLY | 0.7392 | 0.7441 | 0.7444 | 0.7308 | 0.7976 | 0.0529 |
| Logistic Regression | S2_LATE_G1_G2 | 0.8861 | 0.8977 | 0.8952 | 0.8538 | 0.9612 | 0.0369 |
| Random Forest | S0_EARLY_NO_GRADE | 0.4759 | 0.4651 | 0.4595 | 0.5000 | 0.4513 | 0.0772 |
| Random Forest | S1_MID_G1_ONLY | 0.7165 | 0.7241 | 0.7233 | 0.7692 | 0.8218 | 0.0514 |
| Random Forest | S2_LATE_G1_G2 | 0.8911 | 0.8939 | 0.8998 | 0.8923 | 0.9625 | 0.0321 |
| XGBoost | S0_EARLY_NO_GRADE | 0.4911 | 0.4361 | 0.4436 | 0.4231 | 0.4529 | 0.1913 |
| XGBoost | S1_MID_G1_ONLY | 0.7114 | 0.7144 | 0.7212 | 0.6846 | 0.8151 | 0.0885 |
| XGBoost | S2_LATE_G1_G2 | 0.8734 | 0.8785 | 0.8815 | 0.8615 | 0.9500 | 0.0375 |
| MLP | S0_EARLY_NO_GRADE | 0.5367 | 0.4203 | 0.4022 | 0.3538 | 0.4564 | 0.0799 |
| MLP | S1_MID_G1_ONLY | 0.7418 | 0.7393 | 0.7466 | 0.7231 | 0.8252 | 0.0727 |
| MLP | S2_LATE_G1_G2 | 0.8532 | 0.8621 | 0.8595 | 0.8385 | 0.9503 | 0.0797 |

### MLP information gain

| Delta | Macro-F1 | Balanced Accuracy | Low Recall | PR-AUC | ECE |
|---|---:|---:|---:|---:|---:|
| S1-S0 | 0.3445 | 0.3190 | 0.3692 | 0.3687 | -0.0072 |
| S2-S1 | 0.1129 | 0.1228 | 0.1154 | 0.1251 | 0.0070 |
| S2-S0 | 0.4574 | 0.4418 | 0.4846 | 0.4939 | -0.0002 |

## Student-Por

| Model | Scenario | Accuracy | Balanced Accuracy | Macro-F1 | Low Recall | PR-AUC | ECE |
|---|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | S0_EARLY_NO_GRADE | 0.4961 | 0.5878 | 0.5000 | 0.6300 | 0.5110 | 0.1133 |
| Logistic Regression | S1_MID_G1_ONLY | 0.8012 | 0.7945 | 0.7688 | 0.7400 | 0.8286 | 0.0369 |
| Logistic Regression | S2_LATE_G1_G2 | 0.8644 | 0.8581 | 0.8379 | 0.8000 | 0.9148 | 0.0351 |
| Random Forest | S0_EARLY_NO_GRADE | 0.5470 | 0.5641 | 0.5180 | 0.5500 | 0.5117 | 0.0447 |
| Random Forest | S1_MID_G1_ONLY | 0.8259 | 0.8123 | 0.7949 | 0.7600 | 0.8183 | 0.0499 |
| Random Forest | S2_LATE_G1_G2 | 0.8798 | 0.8612 | 0.8514 | 0.7600 | 0.9291 | 0.0321 |
| XGBoost | S0_EARLY_NO_GRADE | 0.6133 | 0.4351 | 0.4498 | 0.2300 | 0.4851 | 0.1204 |
| XGBoost | S1_MID_G1_ONLY | 0.8182 | 0.7723 | 0.7771 | 0.6800 | 0.8239 | 0.0418 |
| XGBoost | S2_LATE_G1_G2 | 0.8952 | 0.8657 | 0.8677 | 0.7600 | 0.9211 | 0.0486 |
| MLP | S0_EARLY_NO_GRADE | 0.6348 | 0.3693 | 0.3433 | 0.1400 | 0.4841 | 0.0277 |
| MLP | S1_MID_G1_ONLY | 0.8012 | 0.7234 | 0.7440 | 0.5700 | 0.8280 | 0.0333 |
| MLP | S2_LATE_G1_G2 | 0.8706 | 0.8190 | 0.8304 | 0.6400 | 0.9147 | 0.0475 |

### MLP information gain

| Delta | Macro-F1 | Balanced Accuracy | Low Recall | PR-AUC | ECE |
|---|---:|---:|---:|---:|---:|
| S1-S0 | 0.4007 | 0.3542 | 0.4300 | 0.3439 | 0.0055 |
| S2-S1 | 0.0864 | 0.0956 | 0.0700 | 0.0867 | 0.0142 |
| S2-S0 | 0.4871 | 0.4497 | 0.5000 | 0.4306 | 0.0198 |

## Claim boundary

S0 is not called CNN-BiLSTM because it has no temporal grade input. The shared MLP and tabular baselines isolate information availability. All selection occurred on inner folds; outer predictions were scored once.
