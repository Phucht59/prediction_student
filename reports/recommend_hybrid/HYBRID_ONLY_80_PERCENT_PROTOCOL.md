# Protocol hoàn thiện module khuyến nghị Hybrid-only

## Kiến trúc được phép

Mô hình học duy nhất trong đường chạy cuối là residual CNN–BiLSTM đã đóng băng. Module khuyến nghị không huấn luyện hoặc gọi XGBoost, LightGBM, LambdaMART, Logistic Regression, pairwise ranker hay một mô hình học máy thứ hai.

Luồng cuối:

```text
Dữ liệu OULAD trước cutoff
→ frozen preprocessing
→ residual CNN–BiLSTM
→ baseline risk và uncertainty
→ deterministic stage-aware actions
→ hybrid counterfactual replay
→ deterministic score và selective abstention
→ prerequisite/conflict/workload/safety constraints
→ recommendation hoặc policy fallback
```

## Định nghĩa “đúng trên 80%”

Chỉ số chính là:

```text
Precision@1 =
Số khuyến nghị top-1 đã phát hành có silver_positive = 1
/
Tổng số khuyến nghị top-1 đã phát hành
```

Silver label được tạo từ hành vi tương lai trực tiếp tương ứng với từng action trong OULAD và chỉ dùng để đánh giá held-out. Silver label không được đưa vào runtime scoring.

Ngưỡng phát hành:

```text
OOF Precision@1 >= 0.80
Actionable coverage >= 0.50
Bootstrap lower 95% Precision@1 >= 0.78
Mỗi outer fold Precision@1 >= 0.75
Mỗi stage đủ support Precision@1 >= 0.70
Action diversity >= 3
Top-action concentration <= 0.75
```

Coverage được tính trên các learner-stage group có ít nhất một action tích cực theo silver label. Vì vậy hệ thống không thể đạt 80% bằng cách abstain gần như toàn bộ trường hợp.

## Candidate action khoa học

```text
ASSESSMENT_COMPLETION
STUDY_REGULARITY → STUDY_SCHEDULE
VLE_ENGAGEMENT
QUIZ_OR_RETRIEVAL_PRACTICE → RETRIEVAL_PRACTICE
CONTENT_REVIEW → LEARNING_CONSOLIDATION
```

Các action liên hệ con người hoặc monitoring không có proxy hành vi trực tiếp được giữ trong safety policy nhưng không nằm trong mẫu số accuracy:

```text
INSTRUCTOR_CONTACT
ADVISOR_ESCALATION
DIAGNOSTIC_CHECK
PROGRESS_MONITORING
```

## Công thức deterministic

```text
score(action) =
  w_risk × normalized hybrid risk reduction
+ w_evidence × semantic evidence strength
+ w_need × ordinal evidence severity
+ w_certainty × hybrid certainty
- w_workload × normalized workload
```

Trước ranking, candidate bị loại nếu không đủ availability, prerequisite, risk reduction, evidence hoặc uncertainty. Sau ranking, hệ thống chỉ phát hành khi top score và top margin đủ lớn; nếu không sẽ abstain/fallback.

Các weight và threshold được chọn bằng nested grouped evaluation trên outer-training folds. Đây là tinh chỉnh công thức cố định, không phải huấn luyện một mô hình xếp hạng.

## Phòng chống leakage

- Cùng sinh viên luôn nằm trong cùng partition.
- Normalization scale chỉ fit trên training partition.
- Threshold chỉ chọn từ inner validation của outer-training.
- Outer-test không được dùng để chỉnh formula hoặc release gate.
- Future signal và silver label không nằm trong runtime features.
- Protected attributes không nằm trong scorer.

## Cách hiểu khoa học

Nếu gate đạt, phát biểu được phép là:

> Trên held-out OULAD trajectories, module hybrid-only đạt Precision@1 tối thiểu 80% trên các khuyến nghị được phát hành, với coverage tối thiểu 50%, theo silver label hành vi tương lai trực tiếp đã đăng ký trước.

Không được phát biểu:

- khuyến nghị chắc chắn giúp tăng điểm;
- hệ thống đã chứng minh tác động nhân quả;
- 80% là độ chính xác trong triển khai thực tế ở mọi trường;
- hệ thống đã được chuyên gia xác nhận;
- hệ thống sẵn sàng production.

Claim boundary:

```text
HYBRID_MODEL_GUIDED_DECISION_SUPPORT_NOT_CAUSAL_EFFECT
```
