# Thesis main results

| Dataset | Model | Macro-F1 | Authority |
|---|---|---:|---|
| Student-Mat | CNN-BiLSTM | 0.901460 | Primary endpoint |
| Student-Por | CNN-BiLSTM | 0.862259 | Primary endpoint |
| OULAD | H0 CNN-BiLSTM | 0.828084 | Legacy endpoint with score-proxy caveat |
| OULAD | H1 Tabular Residual Hybrid | 0.798400 | Strict no-unverified-score endpoint |

## OULAD endpoint authority

| Model | Macro-F1 | Status | Feature-availability protocol |
|---|---:|---:|---:|
| H0 CNN-BiLSTM | 0.828084 | Legacy endpoint | Conservative score proxy |
| H1 Tabular Residual Hybrid | 0.798400 | Strict endpoint | Unverified score values excluded |
| MLP | 0.828286 | Historical comparator | Conservative score proxy |

The two OULAD values must not be presented as results from the same feature-
availability protocol. The stage-aware H1 mean must not be substituted for an
endpoint result.

## Thesis-ready narrative

Trên UCI Student-Mat và Student-Por, mô hình CNN-BiLSTM đạt Macro-F1 lần lượt
0.9015 và 0.8623. Với OULAD, kết quả endpoint lịch sử của CNN-BiLSTM đạt
0.8281 dưới cơ chế score-availability proxy. Khi áp dụng giao thức nghiêm ngặt
hơn, loại bỏ các score-progress feature không chứng minh được thời điểm công
bố, H1 đạt Macro-F1 0.7984. H1 được giữ làm mô hình dự báo sớm vì cả nhánh
temporal và residual đều cho đóng góp dương trong ablation.
