# Locked headline metrics — Hybrid C0-R vs one-weight baselines

Four metrics: **Accuracy**, **Precision** (positive/risk class), **F1**, and the **highest** of AP vs ROC-AUC (here always ROC-AUC).
Thresholds from STOP only. Outer test: **false**.

Serving prediction: **Hybrid CNN–BiLSTM C0** (`src/prediction`). Research comparison below is **C0-R** vs one-weight baselines (Optuna 40/28).
Recommendation V is unchanged (`src/recommend_hybrid/v3`).

## UCI

### Hybrid C0-R

| Mốc | Accuracy | Precision | F1 | Chỉ số cao nhất | Giá trị |
|---|---:|---:|---:|---|---:|
| S0 | 0.6796 | 0.3846 | 0.4149 | roc_auc | 0.6957 |
| S1 | 0.8655 | 0.7240 | 0.7180 | roc_auc | 0.9381 |
| S2 | 0.9031 | 0.7826 | 0.7975 | roc_auc | 0.9699 |

### One-weight baselines (same four)

**LR**

| Mốc | Accuracy | Precision | F1 | Chỉ số cao nhất | Giá trị |
|---|---:|---:|---:|---|---:|
| S0 | 0.7196 | 0.3986 | 0.4014 | roc_auc | 0.6765 |
| S1 | 0.8336 | 0.6466 | 0.6532 | roc_auc | 0.9160 |
| S2 | 0.9052 | 0.7773 | 0.8057 | roc_auc | 0.9628 |

**DT**

| Mốc | Accuracy | Precision | F1 | Chỉ số cao nhất | Giá trị |
|---|---:|---:|---:|---|---:|
| S0 | 0.6831 | 0.4193 | 0.5035 | roc_auc | 0.7083 |
| S1 | 0.8595 | 0.7003 | 0.7146 | roc_auc | 0.9220 |
| S2 | 0.8595 | 0.7003 | 0.7146 | roc_auc | 0.9269 |

**RF**

| Mốc | Accuracy | Precision | F1 | Chỉ số cao nhất | Giá trị |
|---|---:|---:|---:|---|---:|
| S0 | 0.7137 | 0.4469 | 0.5191 | roc_auc | 0.7795 |
| S1 | 0.8745 | 0.7193 | 0.7420 | roc_auc | 0.9298 |
| S2 | 0.8885 | 0.7545 | 0.7704 | roc_auc | 0.9554 |

**SVM**

| Mốc | Accuracy | Precision | F1 | Chỉ số cao nhất | Giá trị |
|---|---:|---:|---:|---|---:|
| S0 | 0.6759 | 0.3729 | 0.4479 | roc_auc | 0.6732 |
| S1 | 0.8247 | 0.6195 | 0.6570 | roc_auc | 0.9094 |
| S2 | 0.8494 | 0.6663 | 0.7071 | roc_auc | 0.9411 |

**XGB**

| Mốc | Accuracy | Precision | F1 | Chỉ số cao nhất | Giá trị |
|---|---:|---:|---:|---|---:|
| S0 | 0.7305 | 0.4473 | 0.4804 | roc_auc | 0.7388 |
| S1 | 0.8444 | 0.6642 | 0.6765 | roc_auc | 0.9174 |
| S2 | 0.8842 | 0.7524 | 0.7601 | roc_auc | 0.9530 |

**CatBoost**

| Mốc | Accuracy | Precision | F1 | Chỉ số cao nhất | Giá trị |
|---|---:|---:|---:|---|---:|
| S0 | 0.7013 | 0.4147 | 0.4639 | roc_auc | 0.7331 |
| S1 | 0.8505 | 0.6868 | 0.6893 | roc_auc | 0.9236 |
| S2 | 0.8792 | 0.7365 | 0.7480 | roc_auc | 0.9479 |

**MLP**

| Mốc | Accuracy | Precision | F1 | Chỉ số cao nhất | Giá trị |
|---|---:|---:|---:|---|---:|
| S0 | 0.6854 | 0.3763 | 0.4240 | roc_auc | 0.6759 |
| S1 | 0.8199 | 0.6286 | 0.6134 | roc_auc | 0.8750 |
| S2 | 0.8828 | 0.7874 | 0.7324 | roc_auc | 0.9414 |

## OULAD

### Hybrid C0-R

| Mốc | Accuracy | Precision | F1 | Chỉ số cao nhất | Giá trị |
|---|---:|---:|---:|---|---:|
| 20pct | 0.6811 | 0.5991 | 0.6741 | roc_auc | 0.7773 |
| 35pct | 0.7498 | 0.6745 | 0.7008 | roc_auc | 0.8308 |
| 50pct | 0.8062 | 0.7675 | 0.7324 | roc_auc | 0.8771 |
| 75pct | 0.8705 | 0.8649 | 0.7919 | roc_auc | 0.9168 |
| 100pct | 0.9051 | 0.9279 | 0.8366 | roc_auc | 0.9370 |

### One-weight baselines (same four)

**LR**

| Mốc | Accuracy | Precision | F1 | Chỉ số cao nhất | Giá trị |
|---|---:|---:|---:|---|---:|
| 20pct | 0.6806 | 0.5939 | 0.6810 | roc_auc | 0.7876 |
| 35pct | 0.7467 | 0.6639 | 0.7026 | roc_auc | 0.8281 |
| 50pct | 0.8070 | 0.7637 | 0.7349 | roc_auc | 0.8754 |
| 75pct | 0.8701 | 0.8863 | 0.7854 | roc_auc | 0.9146 |
| 100pct | 0.9042 | 0.9458 | 0.8318 | roc_auc | 0.9344 |

**DT**

| Mốc | Accuracy | Precision | F1 | Chỉ số cao nhất | Giá trị |
|---|---:|---:|---:|---|---:|
| 20pct | 0.6545 | 0.5727 | 0.6524 | roc_auc | 0.7440 |
| 35pct | 0.7073 | 0.6236 | 0.6587 | roc_auc | 0.7848 |
| 50pct | 0.7717 | 0.7056 | 0.6956 | roc_auc | 0.8446 |
| 75pct | 0.8508 | 0.8225 | 0.7626 | roc_auc | 0.8919 |
| 100pct | 0.8871 | 0.9174 | 0.8012 | roc_auc | 0.9143 |

**RF**

| Mốc | Accuracy | Precision | F1 | Chỉ số cao nhất | Giá trị |
|---|---:|---:|---:|---|---:|
| 20pct | 0.6929 | 0.6254 | 0.6676 | roc_auc | 0.7792 |
| 35pct | 0.7381 | 0.6604 | 0.6915 | roc_auc | 0.8210 |
| 50pct | 0.8065 | 0.7731 | 0.7291 | roc_auc | 0.8733 |
| 75pct | 0.8713 | 0.8862 | 0.7881 | roc_auc | 0.9158 |
| 100pct | 0.9032 | 0.9203 | 0.8343 | roc_auc | 0.9404 |

**SVM**

| Mốc | Accuracy | Precision | F1 | Chỉ số cao nhất | Giá trị |
|---|---:|---:|---:|---|---:|
| 20pct | 0.6684 | 0.5835 | 0.6716 | roc_auc | 0.7722 |
| 35pct | 0.7418 | 0.6569 | 0.6987 | roc_auc | 0.8259 |
| 50pct | 0.8030 | 0.7421 | 0.7368 | roc_auc | 0.8736 |
| 75pct | 0.8659 | 0.8612 | 0.7839 | roc_auc | 0.9147 |
| 100pct | 0.9021 | 0.9312 | 0.8299 | roc_auc | 0.9316 |

**XGB**

| Mốc | Accuracy | Precision | F1 | Chỉ số cao nhất | Giá trị |
|---|---:|---:|---:|---|---:|
| 20pct | 0.6905 | 0.6103 | 0.6814 | roc_auc | 0.7913 |
| 35pct | 0.7496 | 0.6795 | 0.6963 | roc_auc | 0.8282 |
| 50pct | 0.8015 | 0.7419 | 0.7350 | roc_auc | 0.8776 |
| 75pct | 0.8708 | 0.8623 | 0.7932 | roc_auc | 0.9196 |
| 100pct | 0.9037 | 0.9152 | 0.8363 | roc_auc | 0.9422 |

**CatBoost**

| Mốc | Accuracy | Precision | F1 | Chỉ số cao nhất | Giá trị |
|---|---:|---:|---:|---|---:|
| 20pct | 0.6806 | 0.5964 | 0.6795 | roc_auc | 0.7921 |
| 35pct | 0.7559 | 0.6902 | 0.7006 | roc_auc | 0.8309 |
| 50pct | 0.8036 | 0.7483 | 0.7359 | roc_auc | 0.8789 |
| 75pct | 0.8700 | 0.8585 | 0.7928 | roc_auc | 0.9197 |
| 100pct | 0.9029 | 0.9260 | 0.8325 | roc_auc | 0.9400 |

**MLP**

| Mốc | Accuracy | Precision | F1 | Chỉ số cao nhất | Giá trị |
|---|---:|---:|---:|---|---:|
| 20pct | 0.6902 | 0.6095 | 0.6786 | roc_auc | 0.7854 |
| 35pct | 0.7428 | 0.6623 | 0.6964 | roc_auc | 0.8253 |
| 50pct | 0.8029 | 0.7465 | 0.7352 | roc_auc | 0.8747 |
| 75pct | 0.8668 | 0.8410 | 0.7908 | roc_auc | 0.9174 |
| 100pct | 0.9033 | 0.9162 | 0.8353 | roc_auc | 0.9398 |

## Decision

Public prediction on main is **Hybrid CNN–BiLSTM (C0)**. This table locks the research comparison.
Do not open outer test. Recommendation V stays locked.
