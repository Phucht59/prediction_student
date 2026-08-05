# Phạm vi cuối của module khuyến nghị

## Kết luận từ Hybrid-only, V3 và V4

Ba vòng đánh giá held-out cho thấy:

```text
Hybrid-only end-to-end Precision@1: 0.2711
Integrated V3 end-to-end Precision@1: 0.6431
Action-aware V4 end-to-end Precision@1: 0.6589
```

V4 feasibility audit chứng minh rằng ngay cả với perfect action ranking, score family hiện tại chỉ đạt tối đa khoảng `0.6989` Precision@1 tại coverage `0.50`. Vì vậy không được tạo V4.1 threshold tuning và không được tiếp tục tuyên bố mục tiêu end-to-end `0.80` là khả thi với representation/target hiện tại.

## Phạm vi khoa học cuối

Module cuối được định nghĩa là:

```text
Conditional Hybrid Action Ranker
```

Nhiệm vụ:

> Sau khi deterministic policy hoặc quy trình chuyên môn đã quyết định một learner-stage cần được hỗ trợ, module xếp hạng các hành động khoa học phù hợp bằng action head tích hợp trên biểu diễn frozen residual CNN–BiLSTM.

Module không chịu trách nhiệm quyết định:

```text
có nên phát khuyến nghị hay không
```

Phần eligibility phải do:

- deterministic stage-aware policy;
- instructor/advisor workflow;
- hoặc một giao thức eligibility mới được đăng ký và đánh giá riêng.

## Mô hình

```text
Frozen residual CNN–BiLSTM — 160,492 parameters
→ frozen learner-state representation
→ integrated conditional action head
→ ranked action list
→ prerequisite / conflict / workload / safety constraints
```

Không sử dụng XGBoost, LightGBM, LambdaMART, Logistic Regression, Random Forest, SVM hoặc một external learned ranker.

## Estimand

Population đánh giá:

```text
held-out learner-stage groups có ít nhất một silver-positive action
```

Metric chính:

```text
Ranking-only Precision@1
```

Định nghĩa:

```text
Top-ranked valid action có silver_positive = 1
/
Tổng positive evaluation groups
```

Đây là metric chuẩn cho action-ranking submodule. Điều kiện `group có positive action` chỉ là population đánh giá; nó không phải oracle được phép dùng ở runtime.

## Phát biểu được phép nếu release PASS

> Trên các learner-stage group held-out có ít nhất một hành động tích cực theo silver label, action head tích hợp từ biểu diễn residual CNN–BiLSTM xếp một hành động tích cực ở vị trí đầu với Precision@1 vượt ngưỡng đã đăng ký.

## Phát biểu bị cấm

Không được nói:

- toàn hệ thống khuyến nghị đúng trên 80%;
- hệ thống biết chính xác khi nào cần khuyến nghị;
- conditional Precision@1 là end-to-end accuracy;
- hệ thống chứng minh tác động nhân quả;
- khuyến nghị bảo đảm tăng điểm;
- runtime đã được production-authorized.

## Trạng thái song song bắt buộc

Ngay cả khi conditional module PASS, vẫn phải giữ:

```text
CONDITIONAL_RECOMMENDATION_MODULE_COMPLETE
END_TO_END_RECOMMENDATION_SYSTEM_NOT_VALIDATED
runtime_authorized = false
```

Claim boundary:

```text
OFFLINE_CONDITIONAL_ACTION_RANKING_NOT_END_TO_END_OR_CAUSAL_EFFECT
```
