# Protocol khóa trước — hybrid_superiority_v2

Khóa **trước** baseline HPO và Hybrid search. Không đổi target, cohort, stage, hoặc metric sau khi nhìn kết quả.

## Bài toán

- Phân loại nhị phân nguy cơ học tập.
- UCI Combined MAT+POR: `Risk = 1` nếu `G3 < 10`. Nhóm: `global_student_group` (quasi-identity).
- OULAD: `Risk = 1` nếu `final_result ∈ {Fail, Withdrawn}`. Nhóm: `id_student`.
- Không pool hai miền. Một class PyTorch, một topology, hai checkpoint độc lập.
- Một checkpoint UCI score S0/S1/S2. Một checkpoint OULAD score 20/35/50/75/100%. Không model riêng 100%.

## Metric

- Primary: **AP** = `sklearn.metrics.average_precision_score`. Không gọi AP là PR-AUC.
- Trapezoidal PR curve ghi riêng `PR-AUC_trapezoid`.
- Threshold chọn trên STOP, từng stage, áp sang VALID. AP không phụ thuộc threshold.
- Không dùng R²/RMSE cho target nhị phân.

## Availability

- UCI S0: không G1/G2. S1: chỉ G1 trên nhánh sequence. S2: G1 rồi G2.
- G3 không bao giờ là predictor. `absences` cấm.
- G1/G2 **không** vào static/aggregate của Hybrid. Baseline Panel A được G1/G2 như cột tabular ở S1/S2 (cùng thông tin thô, khác tensor).
- OULAD: `event_time < cutoff` và `event_time >= observation_start`. Cấm `final_result`, `score`, `date_unregistration` làm predictor.

## Gate phát triển (inner OOF, không outer)

Warm `W = {UCI:S1, UCI:S2, OULAD:35,50,75,100}`. Cold được phép thua trong guardrail.

1. `Delta_AP(s) > 0` mọi `s ∈ W`.
2. `Delta_AP(s) >= max(0.010, 0.10*(1-AP_B(s)))` mọi `s ∈ W`.
3. Warm macro Hybrid > warm macro từng baseline.
4. Thắng ≥ 4/5 seed preregister mỗi warm stage.
5. Cold: UCI S0 không kém quá 0.05 AP; OULAD 20 không kém quá 0.02; Recall@20% không giảm quá 0.05.
6. Integrity tests pass.
7. Full Hybrid hơn ablation retrain độc lập ≥ 0.005 AP tại UCI S2, OULAD 75/100 và warm macro.

## Confirmation

Chỉ sau freeze architecture/code/feature/search/seeds. Nested, không tune trên outer. Bootstrap cluster theo nhóm, 10_000 lần, comparator = max baseline, Holm, simultaneous LB.

## Ứng viên (không search topology vô hạn)

- C0-R: parallel CNN ∥ BiLSTM, softmax 3-way, capacity giảm.
- C1-R: parallel + residual tabular.
- C2-S: serial CNN→BiLSTM + residual tabular.
- C3-G: serial stage-aware gated residual (ưu tiên).

Một topology cho cả hai dataset. Không `if dataset`.

## Baseline roster (không loại model vì thắng)

LR, DT, RF, SVM, XGB, CatBoost, MLP. Panel A là so sánh chính.

## Seeds

Robust: 42, 1201, 2026. Final: + 3407, 7777. Split seed 42. Outer fold 0 test là firewall phát triển.

## Imbalance

SMOTE/ADASYN không áp lên tensor Hybrid (one-hot + aggregate + sequence). Negative result lịch sử được giữ. Hybrid dùng pos_weight / ranking / focal theo ladder inner.

## Recommendation

Gemini là weak label. Không dùng để gán Risk, chọn kiến trúc, hoặc tuyên bố hiệu quả nhân quả. Quota ≤ 480/500 request/model/ngày.
