# Boost night — OULAD 35/50/75 + UCI, cùng kiến trúc

## Không đổi bài toán
Nhãn OULAD vẫn nhị phân `Fail|Withdrawn`. **Không** tách lớp, **không** hai head.

## Cutoff
- Train + STOP: **35%, 50%, 75%** (bỏ 20%).
- 100% chỉ đánh giá sau (không early-stop).
- UCI: train S0–S2, STOP **S1+S2** (S0 không kéo checkpoint).

## Nhiều chỉ số
STOP không chỉ AP:

`score = AP + 0.3 F1 + 0.15 ROC-AUC + 0.1 Rec + 0.1 Prec − 0.25 ECE`

(macro trên các mốc STOP). Lưu cả bảng AP/F1/Prec/Rec/ECE từng epoch.

## Cùng mô hình UCI + OULAD
Một class `BoostHybrid`: CNN 2 block, kernel **3**, dilation (1,2), cổng 3 nhánh, last-step residual, FiLM `progress`. Khác serving chỉ kernel 3 + last + FiLM — cùng họ lai, khác `lr/dropout/batch` như bản khóa.

## Thêm
- Kênh Δ tuần (cả hai miền, mask-safe).
- Pairwise rank loss λ=0.05.
- FIT-only scale/`pos_weight`, outer firewall, 3×3.

## Máy
AMP FP16, 4 thread, không ưu tiên CPU, nghỉ giữa run, không pin 20%/100%.
