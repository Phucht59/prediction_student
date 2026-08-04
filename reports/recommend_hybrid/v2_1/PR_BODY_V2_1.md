## Mục tiêu

Hoàn thiện module khuyến nghị học tập đa giai đoạn dựa trên risk profile của residual CNN–BiLSTM và ground truth hành vi/kết quả tương lai trong OULAD.

PR vẫn ở trạng thái **Draft**, chưa merge.

## Lịch sử nghiên cứu được giữ nguyên

- Counterfactual V1: engineering PASS, external scientific validation FAILED.
- Outcome-grounded V2: implementation COMPLETE, scientific evidence INCONCLUSIVE.

## Outcome-grounded V2.1

Corrected three-fold OOF family screening đã chạy trên 13,235 learner-course records, 33,912 ranking groups và 84,991 candidate rows.

Kết quả hiện tại:

- NDCG@3: `0.543210`
- Precision@1: `0.573425`
- MAP@3: `0.629832`
- MRR: `0.640127`
- Random mean: `0.491944`
- Random p95: `0.493421`

Cả bốn family đã chạy và LambdaMART được chọn trong cả ba outer folds. Tuy nhiên execution này mới dùng cấu hình đầu tiên của từng family, nên vẫn là **first-configuration family screen**, chưa phải kết quả cuối.

## Corrective code mới

- `run_full_registered_search.py`: chạy toàn bộ 18 cấu hình đã đăng ký và archive family-screen hiện tại.
- `run_exact_negative_controls.py`: dùng đúng selected hyperparameters; không giảm LambdaMART từ 100 trees xuống 10 trees.
- `run_exact_ablation.py`: chạy đủ 10 ablations bằng đúng selected hyperparameters.
- `corrected_release.py`: fail closed nếu thiếu full grid, exact controls, exact ablations, bootstrap hoặc safety evidence.
- `LOCAL_FULL_GRID_COMPLETION_TASK.md`: runbook local cuối cùng.

## Việc còn lại

1. Chạy full 18-configuration nested search.
2. Tính lại hai bootstrap estimands.
3. Chạy đủ 200 exact retrained replicates cho từng mandatory control.
4. Chạy đủ 10 exact ablations.
5. Chạy fail-closed scientific gate.
6. Chỉ tích hợp runtime nếu gate trả `OUTCOME_GROUNDED_V2_1_OFFLINE_VALIDATED`.

## Claim boundary

`OFFLINE_PREDICTIVE_RELEVANCE_NOT_CAUSAL_EFFECT`

## Merge

Merge allowed: **NO**. Giữ PR Draft.
