# Kết quả module khuyến nghị Two-Stage V3 tích hợp Hybrid

## 1. Lý do thay đổi kiến trúc

Hybrid-only deterministic trước đó đạt Precision@1 0,2711. Diagnostic cho thấy lỗi chính nằm ở tầng quyết định có nên phát khuyến nghị hay không, không chỉ ở việc xếp hạng action.

- Stage A cũ — precision: 0.3584
- Stage A cũ — recall: 0.5186
- Stage B cũ — conditional Precision@1: 0.7565
- End-to-end cũ: 0.2711

## 2. Kiến trúc V3

Backbone dự đoán là residual CNN–BiLSTM 160.492 tham số đã đóng băng. Hệ thống tái sử dụng student-state embedding 64 chiều và tabular-expert embedding 32 chiều, sau đó học hai head tích hợp: recommendability và conditional action scoring.

```text
Frozen residual CNN–BiLSTM
→ 64-D student state + 32-D tabular expert
→ Stage A recommendability head
→ Stage B conditional action head
→ selective abstention
→ safety / prerequisite / workload constraints
```

Không sử dụng XGBoost, LightGBM, LambdaMART hoặc một ML ranker tách rời.

## 3. Kết quả held-out OOF

- Groups: 29043
- Learners: 12656
- Positive groups: 9304
- Issued groups: 7105
- Stage A precision: 0.6826
- Stage A recall / positive-group coverage: 0.5213
- Stage B conditional Precision@1: 0.9421
- Stage B Precision@1 trên toàn bộ positive groups: 0.9439
- NDCG@3: 0.9742
- MRR: 0.9698
- End-to-end Precision@1: 0.6431
- Abstention rate: 0.7554
- Action diversity: 4
- Top-action concentration: 0.4006

## 4. Outer-fold stability

- Fold 0: end-to-end P@1=0.6284, coverage=0.5703, conditional P@1=0.9375
- Fold 1: end-to-end P@1=0.6457, coverage=0.4720, conditional P@1=0.9424
- Fold 2: end-to-end P@1=0.6574, coverage=0.5211, conditional P@1=0.9467

## 5. Bootstrap theo sinh viên

- End-to-end Precision@1 95% CI: [0.6327, 0.6540]
- Coverage 95% CI: [0.5110, 0.5324]
- Conditional Precision@1 95% CI: [0.9355, 0.9482]

## 6. Safety và reproducibility

- Verification: `PASS`
- Frozen prediction backbone: `True`
- External ML ranker absent: `True`
- Future/protected features absent: `True`
- Exact numeric replay: `True`
- Exact decision replay: `True`

## 7. Scientific release

- Status: `TWO_STAGE_V3_EVIDENCE_BELOW_GATE`
- Main gates pass: `False`
- Negative controls pass: `False`
- Runtime package ready: `False`
- Runtime authorized: `False`
- Thesis-scope completion: `RECOMMENDATION_MODULE_NOT_COMPLETE`

## 8. Cách hiểu ngưỡng 80%

Chỉ được nói mô hình đạt trên 80% khi chính xác metric held-out được nêu đạt trên 0,80. Conditional Precision@1 không được gọi là độ chính xác end-to-end. Coverage tối thiểu 0,50 vẫn là điều kiện bắt buộc để tránh đạt precision cao bằng cách abstain gần như toàn bộ.

## 9. Giới hạn phát biểu

Đây là bằng chứng predictive relevance ngoại tuyến trên OULAD, không phải bằng chứng tác động nhân quả, không bảo đảm tăng điểm, chưa phải xác nhận chuyên gia và chưa chứng minh khả năng production.

Claim boundary: `OFFLINE_PREDICTIVE_RELEVANCE_NOT_CAUSAL_EFFECT`
