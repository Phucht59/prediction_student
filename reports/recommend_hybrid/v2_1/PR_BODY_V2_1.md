## Mục tiêu

Hoàn thiện module khuyến nghị học tập đa giai đoạn dựa trên risk profile của residual CNN–BiLSTM và ground truth hành vi/kết quả tương lai trong OULAD.

PR vẫn ở trạng thái **Draft**, chưa merge.

## Lịch sử nghiên cứu

- Counterfactual V1: engineering PASS, external scientific validation FAILED.
- Outcome-grounded V2: implementation COMPLETE, scientific evidence INCONCLUSIVE.

## Outcome-grounded V2.1

Family-screen corrected OOF đã chạy trên 13,235 learners, 33,912 ranking groups và 84,991 candidate rows.

- NDCG@3: `0.543210`
- Precision@1: `0.573425`
- MAP@3: `0.629832`
- MRR: `0.640127`
- Random mean: `0.491944`
- Random p95: `0.493421`

Cả bốn model family đã chạy và LambdaMART được chọn trong cả ba folds. Tuy nhiên family-screen mới dùng cấu hình đầu tiên của mỗi family, nên chưa phải kết quả cuối.

## Corrective code mới

- Full 18-configuration preregistered search.
- Exact selected-hyperparameter negative controls.
- Exact selected-hyperparameter ablations.
- Strengthened fail-closed release gate.
- Local runbook: `reports/recommend_hybrid/v2_1/LOCAL_FULL_GRID_COMPLETION_TASK.md`.

## Việc còn lại

1. Full 18-configuration nested search.
2. Recompute both bootstrap estimands.
3. 200 exact retrained replicates cho mỗi mandatory control.
4. Đủ 10 exact ablations.
5. Fail-closed scientific release gate.
6. Runtime chỉ được tích hợp khi gate trả `OUTCOME_GROUNDED_V2_1_OFFLINE_VALIDATED`.

## Claim boundary

`OFFLINE_PREDICTIVE_RELEVANCE_NOT_CAUSAL_EFFECT`

## Merge

Merge allowed: **NO**. Giữ PR Draft.
