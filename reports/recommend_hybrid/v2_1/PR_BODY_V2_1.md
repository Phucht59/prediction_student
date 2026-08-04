## Mục tiêu

Hoàn thiện module khuyến nghị học tập đa giai đoạn dựa trên risk profile của residual CNN–BiLSTM và ground truth hành vi/kết quả tương lai trong OULAD.

PR vẫn ở trạng thái **Draft**, chưa merge.

## Lịch sử nghiên cứu được giữ nguyên

### Counterfactual V1

- Engineering validation: PASS
- External scientific validation: FAILED
- Không xóa hoặc diễn giải lại kết quả thất bại.

### Outcome-grounded V2

- Implementation: COMPLETE
- Scientific evidence: INCONCLUSIVE
- Evaluator defects được lưu như historical evidence.

## Outcome-grounded V2.1

Corrected three-fold OOF execution đã chạy trên 13,235 learner-course records, 33,912 ranking groups và 84,991 candidate rows.

Kết quả family-screen hiện tại:

- NDCG@3: `0.543210`
- Precision@1: `0.573425`
- MAP@3: `0.629832`
- MRR: `0.640127`
- Random mean: `0.491944`
- Random p95: `0.493421`

Cả bốn family đã được thực thi và LambdaMART được chọn trong cả ba outer folds. Tuy nhiên, execution này chỉ chạy cấu hình đầu tiên của từng family; do đó vẫn là **first-configuration family screen**, chưa phải model/hyperparameter selection đầy đủ theo protocol.

## Corrective code đã thêm trực tiếp

- `run_full_registered_search.py`: chạy toàn bộ 18 cấu hình đã đăng ký và archive first-configuration evidence.
- `run_exact_negative_controls.py`: retrain controls bằng đúng selected family/hyperparameters; không giảm 100 trees xuống 10 trees.
- `run_exact_ablation.py`: chạy đủ 10 ablations bằng đúng selected family/hyperparameters.
- `corrected_release.py`: fail closed nếu chưa đủ full grid, 200 exact replicates/control, 10 exact ablations, bootstrap, safety và reproducibility evidence.
- `LOCAL_FULL_GRID_COMPLETION_TASK.md`: runbook local duy nhất cho vòng cuối.

## Việc bắt buộc còn lại

1. Chạy full 18-configuration nested search trong cả ba outer folds.
2. Tính lại group-weighted và learner-weighted learner-cluster bootstrap.
3. Chạy 200 exact retrained replicates cho từng mandatory negative control.
4. Chạy đủ 10 exact ablations.
5. Chạy fail-closed scientific gate.
6. Chỉ tích hợp runtime nếu gate trả `OUTCOME_GROUNDED_V2_1_OFFLINE_VALIDATED`.

## Claim boundary

`OFFLINE_PREDICTIVE_RELEVANCE_NOT_CAUSAL_EFFECT`

Không tuyên bố tác động nhân quả, bảo đảm tăng điểm, expert validation hoặc production readiness.

## Merge

Merge allowed: **NO**. Giữ PR Draft cho đến khi full registered execution và scientific gate hoàn tất.
