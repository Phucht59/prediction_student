# Kết quả module khuyến nghị Two-Stage V4 Action-Aware

## 1. Lý do phát triển V4

V3 đã xếp hạng action rất tốt nhưng end-to-end Precision@1 vẫn bị giới hạn bởi false issue ở recommendability gate. V3 chỉ huấn luyện candidate binary loss trên positive groups, nên action head không bị phạt khi tạo xác suất action cao cho negative groups.

- V3 end-to-end Precision@1: 0.6431
- V3 Stage A precision: 0.6826
- V3 conditional Precision@1: 0.9421

## 2. Kiến trúc V4

Backbone residual CNN–BiLSTM 160.492 tham số và embedding cache được giữ nguyên. V4 chỉ thay đổi hai neural head tích hợp:

```text
Frozen residual CNN–BiLSTM
→ direct recommendability head
→ candidate action head học trên mọi valid candidate
→ masked noisy-OR action recommendability
→ direct/action joint gate
→ stage-specific selective thresholds
→ safety / prerequisite / workload constraints
```

Không sử dụng XGBoost, LightGBM, LambdaMART hoặc external ML ranker.

## 3. Kết quả held-out OOF

- Groups: 29043
- Learners: 12656
- Positive groups: 9304
- Issued groups: 6663
- Stage A precision: 0.6953
- Stage A recall / coverage: 0.4980
- Stage B conditional Precision@1: 0.9476
- Ranking-only Precision@1 trên toàn positive groups: 0.9374
- NDCG@3: 0.9723
- MRR: 0.9669
- End-to-end Precision@1: 0.6589
- Abstention rate: 0.7706
- Action diversity: 4
- Top-action concentration: 0.3928

## 4. Stage A discrimination

- direct: ROC-AUC=0.8109, AP=0.6680, Brier=0.1778
- action_derived: ROC-AUC=0.8070, AP=0.6471, Brier=0.2094
- joint: ROC-AUC=0.8084, AP=0.6527, Brier=0.2026

## 5. Outer-fold stability

- Fold 0: end-to-end P@1=0.6565, coverage=0.5218, conditional P@1=0.9403
- Fold 1: end-to-end P@1=0.6628, coverage=0.4623, conditional P@1=0.9524
- Fold 2: end-to-end P@1=0.6578, coverage=0.5095, conditional P@1=0.9506

## 6. Per-stage

- EARLY_20: end-to-end P@1=0.6453, coverage=0.2320, conditional P@1=0.9790
- EARLY_35: end-to-end P@1=0.6415, coverage=0.5385, conditional P@1=0.9315
- MIDDLE_50: end-to-end P@1=0.6764, coverage=0.5697, conditional P@1=0.9560

## 7. Per-action

- ASSESSMENT_COMPLETION: issued=0, precision=0.0000, conditional precision=0.0000
- STUDY_REGULARITY: issued=1288, precision=0.5497, conditional precision=0.9100
- VLE_ENGAGEMENT: issued=2617, precision=0.6974, conditional precision=0.9656
- QUIZ_OR_RETRIEVAL_PRACTICE: issued=373, precision=0.6220, conditional precision=0.8889
- CONTENT_REVIEW: issued=2385, precision=0.6813, conditional precision=0.9536

## 8. Bootstrap theo sinh viên

- End-to-end Precision@1 95% CI: [0.6473, 0.6701]
- Coverage 95% CI: [0.4881, 0.5086]
- Conditional Precision@1 95% CI: [0.9413, 0.9537]

## 9. Safety và reproducibility

- Verification: `PASS`
- Frozen prediction backbone: `True`
- All-group candidate supervision: `True`
- External ML ranker absent: `True`
- Future/protected features absent: `True`
- Exact numeric replay: `True`
- Exact decision replay: `True`

## 10. Scientific release

- Status: `TWO_STAGE_V4_EVIDENCE_BELOW_GATE`
- Main gates pass: `False`
- Negative controls pass: `False`
- Runtime authorized: `False`
- Thesis-scope completion: `RECOMMENDATION_MODULE_NOT_COMPLETE`

## 11. Giới hạn phát biểu

Conditional Precision@1 không được gọi là độ chính xác end-to-end. Kết quả là predictive relevance ngoại tuyến trên OULAD, không phải tác động nhân quả, không bảo đảm tăng điểm, chưa phải xác nhận chuyên gia và chưa chứng minh production readiness.

Claim boundary: `OFFLINE_PREDICTIVE_RELEVANCE_NOT_CAUSAL_EFFECT`
