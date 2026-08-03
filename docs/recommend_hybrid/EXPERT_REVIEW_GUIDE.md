# Hướng dẫn expert review counterfactual recommender

## Phạm vi

Đánh giá 160 case được lấy mẫu xác định từ full cohort. Case chỉ chứa tín hiệu quan sát trước cutoff, candidate actions và ước lượng rủi ro của model. Không có outcome tương lai, protected attributes hoặc mã định danh người học trong gói review.

## Cách review

1. Đọc `EXPERT_REVIEW_CASES.csv` hoặc JSON và đối chiếu `case_id` với các candidate actions.
2. Chấm 10 tiêu chí trong `EXPERT_REVIEW_RUBRIC.csv` theo thang 1–5.
3. Chọn `ACCEPT`, `ACCEPT_WITH_MODIFICATION`, `REJECT` hoặc `ESCALATE_TO_HUMAN`.
4. Ghi hành động sửa đổi và lý do trong `EXPERT_REVIEW_RESULTS_TEMPLATE.csv`.

Không diễn giải `estimated_risk_reduction` là hiệu quả nhân quả hay outcome thực tế. Fallback/abstain là một kết quả hợp lệ cần được đánh giá riêng.
