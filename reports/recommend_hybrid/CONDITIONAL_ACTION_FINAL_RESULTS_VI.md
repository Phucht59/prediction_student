# Kết quả cuối module xếp hạng hành động khuyến nghị có điều kiện

## 1. Ranh giới module

Module này chỉ xếp hạng hành động sau khi một policy xác định trước hoặc quy trình giảng viên/cố vấn đã quyết định rằng sinh viên cần được hỗ trợ. Module không quyết định có nên phát khuyến nghị hay không.

Population đánh giá gồm các learner-stage group held-out có ít nhất một hành động tích cực theo silver label. Điều kiện này chỉ được dùng để đánh giá khả năng xếp hạng, không phải oracle có sẵn trong runtime.

## 2. Kiến trúc

```text
Frozen residual CNN–BiLSTM representation
→ integrated conditional action head
→ ranked scientific actions
→ prerequisite / conflict / workload / safety constraints
```

Không sử dụng XGBoost, LightGBM, LambdaMART, Logistic Regression hoặc một external ML ranker khác.

## 3. Kết quả held-out

- Positive evaluation groups: 9.304
- Positive-population learners: 6.874
- Ranking-only Precision@1: 0,9374
- Bootstrap 95% CI: [0,9325; 0,9422]
- NDCG@3: 0,9723
- MRR: 0,9669
- Action-selection diversity: 4
- Top-action concentration: 0,3600

## 4. Độ ổn định theo outer fold

- Fold 0 Precision@1: 0,9368
- Fold 1 Precision@1: 0,9375
- Fold 2 Precision@1: 0,9380

## 5. Độ ổn định theo giai đoạn

- EARLY_20 Precision@1: 0,9147
- EARLY_35 Precision@1: 0,9306
- MIDDLE_50 Precision@1: 0,9529

## 6. So sánh baseline

- Risk-reduction-only Precision@1: 0,5545
- Evidence-only Precision@1: 0,6480
- Lowest-workload Precision@1: 0,7075
- Mức cải thiện so với baseline tốt nhất: +0,2299

## 7. Negative controls

- Random ranking: 5.000 repetitions, p-value = 0,0002
- Fold/stage-stratified label permutation: 5.000 repetitions, p-value = 0,0002
- Non-identity action permutation: 5.000 repetitions, p-value = 0,0002
- Identity permutation đã được loại khỏi action-identity control.

## 8. Scientific release

```text
CONDITIONAL_ACTION_RANKING_OFFLINE_VALIDATED
CONDITIONAL_RECOMMENDATION_MODULE_COMPLETE
END_TO_END_RECOMMENDATION_SYSTEM_NOT_VALIDATED
```

- Conditional gates: PASS
- Runtime authorized: false
- Merge allowed: NO
- Claim boundary: `OFFLINE_CONDITIONAL_ACTION_RANKING_NOT_END_TO_END_OR_CAUSAL_EFFECT`

## 9. Context end-to-end không thuộc release này

- V4 end-to-end Precision@1: 0,6589
- V4 positive-group coverage: 0,4980

Không được gọi Ranking-only Precision@1 = 0,9374 là độ chính xác toàn bộ hệ thống. Kết quả conditional không chứng minh khả năng tự quyết định khi nào nên phát khuyến nghị.

## 10. Phát biểu được phép

Trên các learner-stage group held-out có ít nhất một hành động tích cực theo silver label, action head tích hợp từ biểu diễn residual CNN–BiLSTM xếp một hành động tích cực ở vị trí đầu với Precision@1 = 93,74%, bootstrap 95% CI [93,25%; 94,22%].

Kết quả không chứng minh tác động nhân quả, bảo đảm tăng điểm, expert validation, production readiness hoặc khả năng tự động xác định thời điểm phát khuyến nghị.
