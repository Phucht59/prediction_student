# STAT_SIGNIFICANCE

Nguồn Hybrid 9-run: `test_lab/artifacts/hybrid_vnext/phase4/ROBUST_CONFIRMATION.csv` (L1_control).
Nguồn baseline: `BASELINE_INNER_RESULTS.csv`.
Kiểm định: Wilcoxon signed-rank trên 9 cặp (fold, seed), bootstrap 2000 trên hiệu AP. **Không** mở outer.

| domain | stage | vs | Hybrid AP | other AP | Δ | 95% CI | Wilcoxon p | Hybrid>other |
|---|---|---|---:|---:|---:|---|---:|---:|
| uci | S0 | LR | 0.4547 | 0.4754 | -0.0206 | [-0.0500, +0.0069] | 0.1641 | 2/9 |
| uci | S0 | RF | 0.4547 | 0.4995 | -0.0447 | [-0.0805, -0.0105] | 0.07422 | 3/9 |
| uci | S1 | LR | 0.8214 | 0.7794 | +0.0420 | [+0.0206, +0.0673] | 0.003906 | 9/9 |
| uci | S1 | RF | 0.8214 | 0.7895 | +0.0319 | [+0.0207, +0.0429] | 0.003906 | 9/9 |
| uci | S2 | LR | 0.9101 | 0.8812 | +0.0289 | [+0.0150, +0.0458] | 0.007812 | 8/9 |
| uci | S2 | RF | 0.9101 | 0.9072 | +0.0029 | [-0.0031, +0.0084] | 0.1641 | 7/9 |
| oulad | 20pct | LR | 0.7624 | 0.7632 | -0.0009 | [-0.0031, +0.0016] | 0.5703 | 3/9 |
| oulad | 20pct | RF | 0.7624 | 0.7522 | +0.0102 | [+0.0072, +0.0133] | 0.003906 | 9/9 |
| oulad | 35pct | LR | 0.8058 | 0.7986 | +0.0073 | [+0.0057, +0.0090] | 0.003906 | 9/9 |
| oulad | 35pct | RF | 0.8058 | 0.7940 | +0.0119 | [+0.0112, +0.0125] | 0.003906 | 9/9 |
| oulad | 50pct | LR | 0.8483 | 0.8399 | +0.0084 | [+0.0074, +0.0093] | 0.003906 | 9/9 |
| oulad | 50pct | RF | 0.8483 | 0.8402 | +0.0081 | [+0.0065, +0.0096] | 0.003906 | 9/9 |
| oulad | 75pct | LR | 0.8885 | 0.8828 | +0.0057 | [+0.0048, +0.0066] | 0.003906 | 9/9 |
| oulad | 75pct | RF | 0.8885 | 0.8847 | +0.0037 | [+0.0012, +0.0062] | 0.05469 | 6/9 |
| oulad | 100pct | LR | 0.9204 | 0.9114 | +0.0090 | [+0.0081, +0.0100] | 0.003906 | 9/9 |
| oulad | 100pct | RF | 0.9204 | 0.9154 | +0.0050 | [+0.0031, +0.0068] | 0.003906 | 9/9 |

## Hai số AP UCI S1: 0.821 vs 0.811

- Serving 3×3 (`uci_final.csv` cột `pr_auc`): S0=0.454744821940755, S1=0.8214149119441972, S2=0.9101038055944976.
- Cùng file OVERFIT_AUDIT / Chương 4 khóa: S1 **0.8214**.
- Robust L1_control mean S1 from ROBUST_CONFIRMATION: **0.8214** (should match serving if same 9 jobs).
- Tensor-parity CSV `baseline_fair_stage_metrics_uci.csv` models=['CatBoost', 'DT', 'LR', 'MLP', 'RF', 'SVM', 'XGB']. **Không có hàng Hybrid** — số 0.811 không thể lấy từ file này.
- Số **0.8110 / 0.9132** được hard-code trong `generate_ch4_figures.py` (`hybrid = {S0: 0.4559, S1: 0.8110, S2: 0.9132}`), comment 'from CHUONG_3 3.3.6'. Đó là panel tensor-parity cũ, **không** phải 9-run serving.
- **Số khóa để báo cáo serving:** UCI S1 AP **0.8214** (9-run L1). Số 0.811 chỉ dùng khi nói panel cùng-tensor (và phải ghi nguồn hard-code / thiếu hàng Hybrid trong CSV).

## Đọc theo claim Hybrid (mốc thiết kế)

Ưu thế có ý nghĩa (p < 0.05, đa số 9/9): **UCI S1 vs LR/RF**, **UCI S2 vs LR**, **OULAD 20% vs RF**, **OULAD 35/50/100% vs LR và RF**, **OULAD 75% vs LR**.

S0 không có chuỗi — không dùng để so kiến trúc lai. OULAD 20% vs LR: Δ −0.0009, CI chứa 0 (tương đương trên mốc lạnh). UCI S2 vs RF: Hybrid vẫn cao hơn điểm (0.910 vs 0.907, 7/9 run); CI Δ hẹp.

Số khóa serving UCI S1 = **0.8214**, không dùng 0.811 trên tài liệu công khai.
