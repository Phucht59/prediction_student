# SUMMARY — kltn science fix (research)

| Mục | Trạng thái | Output |
|---|---|---|
| P0.1 Ablation | TODO GPU | `ABLATION.md`, `artifacts/research/kltn_science_fix/ablation_raw.csv` |
| P0.2 Wilcoxon + dual AP | XONG | `STAT_SIGNIFICANCE.md` |
| P0.3 Fail/Withdrawn | XONG | `LABEL_SPLIT_ANALYSIS.md` |
| P0.4 Fairness | XONG | `FAIRNESS_BY_GROUP.md`, `figures/fairness_ap_by_group_35pct.png` |
| P0.5 Gate | XONG | `GATE_WEIGHTS.md`, `figures/gate_weights_by_cutoff.png` |
| P0.6 Ch5 | XONG bản research | `chapters/CHUONG_5.md` |
| P1 PR/ROC/CM/reliability | XONG | `figures/pr_*.png` `roc_*` `confusion_*` `reliability_*` |
| P1 Spearman FIT | XONG | `SPEARMAN_FIT.md` |
| P1 Survivorship | XONG | `SURVIVAL.md` |
| P2 Ch1 Ch2 | XONG bản research | `chapters/CHUONG_1.md` `CHUONG_2.md` |
| Ch3/Ch4 bổ sung | XONG bản research | `chapters/CHUONG_3.md` `CHUONG_4.md` |
| Hình kiến trúc | XONG | `figures/architecture_hybrid.png` |
| Outer test | **Không mở** | — |
| Giao diện / FastAPI | **Không làm** (phạm vi = mô hình) | Ch5 hướng phát triển |

## Mâu thuẫn số liệu đã giải quyết

Serving UCI S1 AP **0.8214** = ROBUST_CONFIRMATION L1. Số **0.811** không có trong CSV tensor-parity (thiếu hàng Hybrid), chỉ hard-code trong `generate_ch4_figures.py`.

## Mâu thuẫn / việc chưa merge main

Toàn bộ nằm `reports/research/hybrid_superiority_v2/` và `research/kltn_science_fix/`. User duyệt rồi mới đưa vào Chương nộp.
