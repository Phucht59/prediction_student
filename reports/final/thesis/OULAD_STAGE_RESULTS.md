# OULAD stage-aware results

## Frozen H1 early-warning authority

| Stage | Observed | Accuracy | Balanced Accuracy | Macro Precision | Macro Recall | Macro-F1 | PR-AUC | ROC-AUC | NLL | Brier | ECE | Risk Precision | Risk Recall | Risk F1 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| E1 | 20% | 0.719426 | 0.712568 | 0.715645 | 0.712568 | 0.713635 | 0.772028 | 0.784488 | 0.550339 | 0.186649 | 0.053802 | 0.692879 | 0.654166 | 0.672944 |
| E2 | 35% | 0.760635 | 0.747961 | 0.756378 | 0.747961 | 0.750632 | 0.816099 | 0.830836 | 0.484385 | 0.160384 | 0.031334 | 0.736049 | 0.670644 | 0.701246 |
| M1 | 50% | 0.806263 | 0.789399 | 0.802225 | 0.789399 | 0.793953 | 0.861498 | 0.875771 | 0.407844 | 0.131468 | 0.017606 | 0.785856 | 0.707135 | 0.743818 |
| L1 | 75% | 0.866352 | 0.838451 | 0.873037 | 0.838451 | 0.850333 | 0.906090 | 0.919401 | 0.313123 | 0.096265 | 0.013618 | 0.886700 | 0.733528 | 0.801550 |

One estimator/checkpoint per fold and seed serves all four cutoffs. The mean across stages is not an endpoint metric.

## Frozen comparator tables

Protocol IDs remain visible because classical comparators and the Phase 6 H1/MLP pair come from separate frozen evidence namespaces.

### E1 — 20% observed

| Model | Protocol | Accuracy | Balanced Accuracy | Macro-F1 | PR-AUC | ROC-AUC | NLL | Brier | ECE |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Decision Tree | unified_stage_aware_oulad_v2 | 0.666791 | 0.631551 | 0.612486 | 0.684930 | 0.716772 | 0.968072 | 0.229474 | 0.128201 |
| H1 Tabular Residual Hybrid | h1_final_outer_v1 | 0.719426 | 0.712568 | 0.713635 | 0.772028 | 0.784488 | 0.550339 | 0.186649 | 0.053802 |
| HistGradientBoosting | unified_stage_aware_oulad_v2 | 0.721054 | 0.701114 | 0.702010 | 0.774482 | 0.785465 | 0.542415 | 0.182920 | 0.018630 |
| Logistic Regression | unified_stage_aware_oulad_v2 | 0.716827 | 0.697466 | 0.698417 | 0.758977 | 0.776752 | 0.563843 | 0.191752 | 0.062624 |
| MLP | h1_final_outer_v1 | 0.721353 | 0.709697 | 0.711734 | 0.771220 | 0.783174 | 0.547452 | 0.184459 | 0.022276 |
| Random Forest | unified_stage_aware_oulad_v2 | 0.716101 | 0.696189 | 0.696941 | 0.760821 | 0.777376 | 0.554715 | 0.187916 | 0.032192 |
| SVM | unified_stage_aware_oulad_v2 | 0.719664 | 0.701729 | 0.703243 | 0.752423 | 0.775255 | 0.564055 | 0.190269 | 0.050898 |
| XGBoost | unified_stage_aware_oulad_v2 | 0.723824 | 0.705412 | 0.706965 | 0.775968 | 0.787472 | 0.540147 | 0.182222 | 0.017181 |

### E2 — 35% observed

| Model | Protocol | Accuracy | Balanced Accuracy | Macro-F1 | PR-AUC | ROC-AUC | NLL | Brier | ECE |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Decision Tree | unified_stage_aware_oulad_v2 | 0.721121 | 0.687892 | 0.689896 | 0.711969 | 0.755733 | 0.986081 | 0.208019 | 0.120949 |
| H1 Tabular Residual Hybrid | h1_final_outer_v1 | 0.760635 | 0.747961 | 0.750632 | 0.816099 | 0.830836 | 0.484385 | 0.160384 | 0.031334 |
| HistGradientBoosting | unified_stage_aware_oulad_v2 | 0.766461 | 0.747579 | 0.752369 | 0.817004 | 0.830408 | 0.481615 | 0.159372 | 0.016618 |
| Logistic Regression | unified_stage_aware_oulad_v2 | 0.757739 | 0.740246 | 0.744397 | 0.802585 | 0.822153 | 0.513858 | 0.171558 | 0.081838 |
| MLP | h1_final_outer_v1 | 0.764082 | 0.744668 | 0.749499 | 0.816127 | 0.829741 | 0.483335 | 0.159755 | 0.019252 |
| Random Forest | unified_stage_aware_oulad_v2 | 0.756680 | 0.737717 | 0.742204 | 0.807823 | 0.823316 | 0.493327 | 0.163975 | 0.032516 |
| SVM | unified_stage_aware_oulad_v2 | 0.761626 | 0.743734 | 0.748140 | 0.804476 | 0.824333 | 0.497373 | 0.163947 | 0.028807 |
| XGBoost | unified_stage_aware_oulad_v2 | 0.764337 | 0.748506 | 0.752427 | 0.818223 | 0.832404 | 0.479605 | 0.158763 | 0.017885 |

### M1 — 50% observed

| Model | Protocol | Accuracy | Balanced Accuracy | Macro-F1 | PR-AUC | ROC-AUC | NLL | Brier | ECE |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Decision Tree | unified_stage_aware_oulad_v2 | 0.774872 | 0.747529 | 0.754699 | 0.769990 | 0.807797 | 0.847980 | 0.175801 | 0.106630 |
| H1 Tabular Residual Hybrid | h1_final_outer_v1 | 0.806263 | 0.789399 | 0.793953 | 0.861498 | 0.875771 | 0.407844 | 0.131468 | 0.017606 |
| HistGradientBoosting | unified_stage_aware_oulad_v2 | 0.803934 | 0.791659 | 0.793757 | 0.860887 | 0.874499 | 0.407842 | 0.131718 | 0.014846 |
| Logistic Regression | unified_stage_aware_oulad_v2 | 0.798337 | 0.787441 | 0.788617 | 0.851862 | 0.868742 | 0.429583 | 0.139131 | 0.052335 |
| MLP | h1_final_outer_v1 | 0.810700 | 0.787970 | 0.795505 | 0.861969 | 0.875541 | 0.407380 | 0.131274 | 0.014103 |
| Random Forest | unified_stage_aware_oulad_v2 | 0.799636 | 0.788281 | 0.789743 | 0.856090 | 0.870630 | 0.416893 | 0.134096 | 0.024336 |
| SVM | unified_stage_aware_oulad_v2 | 0.801527 | 0.791393 | 0.792235 | 0.852322 | 0.870357 | 0.423162 | 0.134353 | 0.024666 |
| XGBoost | unified_stage_aware_oulad_v2 | 0.800487 | 0.790247 | 0.791110 | 0.862133 | 0.875952 | 0.405911 | 0.131004 | 0.016190 |

### L1 — 75% observed

| Model | Protocol | Accuracy | Balanced Accuracy | Macro-F1 | PR-AUC | ROC-AUC | NLL | Brier | ECE |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Decision Tree | unified_stage_aware_oulad_v2 | 0.817621 | 0.805858 | 0.804592 | 0.846076 | 0.874075 | 0.602358 | 0.128669 | 0.081170 |
| H1 Tabular Residual Hybrid | h1_final_outer_v1 | 0.866352 | 0.838451 | 0.850333 | 0.906090 | 0.919401 | 0.313123 | 0.096265 | 0.013618 |
| HistGradientBoosting | unified_stage_aware_oulad_v2 | 0.837443 | 0.835587 | 0.828444 | 0.906477 | 0.919366 | 0.312450 | 0.096326 | 0.014055 |
| Logistic Regression | unified_stage_aware_oulad_v2 | 0.834849 | 0.831440 | 0.825288 | 0.900175 | 0.913450 | 0.329665 | 0.102403 | 0.036626 |
| MLP | h1_final_outer_v1 | 0.870525 | 0.839522 | 0.853659 | 0.907916 | 0.921283 | 0.309635 | 0.095301 | 0.015475 |
| Random Forest | unified_stage_aware_oulad_v2 | 0.839904 | 0.836156 | 0.830438 | 0.903773 | 0.916743 | 0.318962 | 0.098379 | 0.027565 |
| SVM | unified_stage_aware_oulad_v2 | 0.841541 | 0.838813 | 0.832425 | 0.897031 | 0.910648 | 0.339809 | 0.100420 | 0.044136 |
| XGBoost | unified_stage_aware_oulad_v2 | 0.840933 | 0.838966 | 0.832026 | 0.907662 | 0.920791 | 0.310244 | 0.095650 | 0.015595 |
