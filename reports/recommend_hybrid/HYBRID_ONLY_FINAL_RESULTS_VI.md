# Kết quả cuối module khuyến nghị Hybrid-only

## 1. Kiến trúc

Mô hình học duy nhất là residual CNN–BiLSTM đã đóng băng. Phần khuyến nghị sử dụng policy theo giai đoạn, mô phỏng phản thực bằng chính mô hình hybrid, công thức utility cố định và selective abstention. Không sử dụng XGBoost, LightGBM, LambdaMART hoặc một mô hình xếp hạng học máy thứ hai.

## 2. Cách hiểu ngưỡng 80%

Ngưỡng 80% là Precision@1 trên các khuyến nghị thực sự được phát hành, so với silver label hành vi tương lai trực tiếp trong OULAD. Đây không phải tỷ lệ bảo đảm sinh viên tăng điểm và không phải bằng chứng nhân quả.

## 3. Cohort đánh giá

- Transition groups: 45953
- Rankable groups: 29043
- Candidate rows: 82847
- Groups có ít nhất một action tích cực tương lai: 9304

## 4. Kết quả OOF

- Precision@1: 0.2711
- Actionable coverage: 0.5186
- Issued groups: 13462
- Action diversity: 4
- Top-action concentration: 0.4816

## 5. Bootstrap theo sinh viên

- Precision@1 95% CI: [0.2638, 0.2787]
- Coverage 95% CI: [0.5085, 0.5291]

## 6. Outer-fold stability

- Fold 0: Precision@1=0.2796, coverage=0.5398
- Fold 1: Precision@1=0.2760, coverage=0.5287
- Fold 2: Precision@1=0.2571, coverage=0.4873

## 7. Deterministic baselines

- evidence_only: Precision@1=0.2337, coverage=0.7417
- lowest_workload: Precision@1=0.2366, coverage=0.7417
- risk_reduction_only: Precision@1=0.2379, coverage=0.7417

## 8. Safety và reproducibility

- Verification status: PASS
- Temporal leakage: True
- Protected-feature exclusion: True
- Deterministic replay: True

## 9. Scientific release

- Status: `HYBRID_ONLY_SILVER_EVIDENCE_BELOW_GATE`
- Thesis-scope completion: `RECOMMENDATION_MODULE_NOT_COMPLETE`
- Runtime authorized: `False`
- Claim boundary: `HYBRID_MODEL_GUIDED_DECISION_SUPPORT_NOT_CAUSAL_EFFECT`

## 10. Giới hạn phát biểu

Kết quả chỉ cho phép nói hệ thống chuyển dự đoán rủi ro của hybrid thành khuyến nghị minh bạch và đạt mức phù hợp nhất định với hành vi tương lai quan sát được. Không được nói hệ thống chứng minh tác động nhân quả, bảo đảm tăng điểm, đã được chuyên gia xác nhận hoặc sẵn sàng production.
