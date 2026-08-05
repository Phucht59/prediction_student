# Kết quả cuối module xếp hạng hành động khuyến nghị có điều kiện

## Ranh giới module

Module này chỉ xếp hạng hành động sau khi một policy hoặc quy trình con người đã quyết định cần hỗ trợ. Module không quyết định có nên phát khuyến nghị hay không.

## Kết quả held-out

- Ranking-only Precision@1: 0.9374
- Bootstrap 95% CI: [0.9325, 0.9422]
- NDCG@3: 0.9723
- MRR: 0.9669
- Positive evaluation groups: 9304
- Action diversity: 4

## Context end-to-end không thuộc release này

- V4 end-to-end Precision@1: 0.6589
- V4 positive-group coverage: 0.4980
- Không được gọi conditional Precision@1 là độ chính xác toàn hệ thống.

## Baseline

- risk_reduction_only: Precision@1=0.5545
- evidence_strength_only: Precision@1=0.6480
- lowest_workload: Precision@1=0.7075

## Controls

- Random ranking p-value: 0.000200
- Label permutation p-value: 0.000200
- Action identity permutation p-value: 0.000200

## Scientific status

- Conditional module: `CONDITIONAL_ACTION_RANKING_OFFLINE_VALIDATED`
- Thesis scope: `CONDITIONAL_RECOMMENDATION_MODULE_COMPLETE`
- End-to-end system: `END_TO_END_RECOMMENDATION_SYSTEM_NOT_VALIDATED`
- Runtime authorized: `false`
- Claim boundary: `OFFLINE_CONDITIONAL_ACTION_RANKING_NOT_END_TO_END_OR_CAUSAL_EFFECT`

## Phát biểu được phép

Trên các learner-stage group held-out có ít nhất một hành động tích cực theo silver label, action head tích hợp từ biểu diễn residual CNN–BiLSTM xếp một hành động tích cực ở vị trí đầu với Precision@1 được báo cáo ở trên.

Kết quả không chứng minh khả năng quyết định khi nào nên phát khuyến nghị, tác động nhân quả, bảo đảm tăng điểm hoặc production readiness.
