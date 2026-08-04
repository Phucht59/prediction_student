## Mục tiêu

Hoàn thiện module khuyến nghị học tập đa giai đoạn dựa trên risk profile của residual CNN–BiLSTM và ground truth hành vi/kết quả tương lai trong OULAD.

PR vẫn ở trạng thái **Draft**, chưa merge.

## Lịch sử nghiên cứu

- Counterfactual V1: engineering PASS, external scientific validation FAILED.
- Outcome-grounded V2: implementation COMPLETE, scientific evidence INCONCLUSIVE.

## Outcome-grounded V2.1 full registered grid

Full 18-configuration search đã hoàn tất trên cả ba outer folds:

- 18/18 trial COMPLETE tại mỗi outer fold;
- selected model cả ba folds: LambdaMART/XGBoost;
- selected parameters: `learning_rate=0.1`, `n_estimators=250`, `num_leaves=15`;
- learners: 13,235;
- ranking groups: 33,912;
- candidate rows: 84,991.

Corrected OOF results:

- NDCG@3: `0.559664`;
- Precision@1: `0.607366`;
- MAP@3: `0.650578`;
- MRR: `0.658769`;
- random mean: `0.491944`;
- random p95: `0.493421`;
- policy baseline NDCG@3: `0.496025`;
- counterfactual V1 NDCG@3: `0.485995`.

Learner-cluster bootstrap đã hoàn tất 2,000 replicates cho cả group-weighted và learner-weighted estimands. Group-weighted improvement so với policy là `+0.060333`, CI 95% `[0.058054, 0.062519]`.

## Scientific execution còn lại

Full-grid và bootstrap đã COMPLETE, nhưng scientific release chưa được phép vì:

1. Sáu authority-bound retrained negative controls chưa đủ 200 replicates/control.
2. Mười authority-bound ablations chưa chạy đủ.
3. Release gate vẫn fail-closed.

Đã bổ sung `run_parallel_authority_controls.py` để chạy các batch độc lập song song mà không giảm 200 replicates, tree count, depth, feature set hoặc outer folds. Mỗi model fit dùng một CPU thread và batch outputs vẫn resume-safe/model-authority-bound.

## Claim boundary

`OFFLINE_PREDICTIVE_RELEVANCE_NOT_CAUSAL_EFFECT`

Không tuyên bố tác động nhân quả, bảo đảm tăng điểm, expert validation hoặc production readiness.

## Runtime và merge

Runtime chỉ được tích hợp nếu gate tạo:

`OUTCOME_GROUNDED_V2_1_OFFLINE_VALIDATED`

Merge allowed: **NO**. Giữ PR Draft.
