# Kết luận Strategy B Phase A-B — `strategy-b-phase-ab-20260714-475a672`

## 1. Correctness đã sửa

- Canonical resolved configuration giữ cả suggested parameters và fixed constants.
- Missing required key fail-fast; không còn silent balanced-class-weight fallback.
- Inner, outer và final refit dùng chung `StudentEstimatorFactory` và một training-partition factory.
- B1/B2 dùng fixed learning rate, không SWA; epoch-selection và full refit replay cùng policy.
- Final estimator path refit trên toàn development frame.
- PostgreSQL loader của selection chỉ truy vấn 316 development source rows.
- Checkpoints và preprocessors được hash và tái tạo exact OOF probabilities.

## 2. Control reproduction và B1/B2

| Policy | Mean fold-seed Macro-F1 | SD | Mean accuracy |
|---|---:|---:|---:|
| A0_historical_compatible | 0.801208 | 0.085439 | 0.817560 |
| B1_drop_last_true | 0.835214 | 0.065800 | 0.844891 |
| B2_drop_last_false | 0.832013 | 0.094833 | 0.841667 |

Paired B2−B1 Macro-F1 mean: **-0.003201**; wins/ties/losses: **8/1/6** trên 15 fold-seed pairs.

A0 là historical-compatible diagnostic control và cố ý giữ scheduler/SWA mismatch để đo chênh lệch; không đủ điều kiện làm estimator chính thức. B1/B2 là corrected estimators.

## 3. Validation

- Full test suite: **PASS** (`return_code=0`).
- Estimator parity B1/B2: **PASS**.
- Exact checkpoint reproduction: **PASS**, max abs diff `0`.
- Strict validation: **PASS**.
- 79 observed payload accessed: **NO**.

## 4. Vấn đề còn lại

- Phase C candidate comparison chưa chạy; không có best_overall_model hoặc best_thesis_hybrid_model.
- Ordinal, residual và multitask candidates chưa được triển khai.
- Recommendation Level 0.5 chưa được sửa trong Phase A-B.
- Không có unseen external confirmation dataset; 79 records vẫn bị quarantine.
- Phase C vẫn cần phê duyệt riêng dù technical gate có thể PASS.

## 5. Phase C gate

Technical prerequisites: **ĐẠT**.

Phase C **không được tự động mở**. Cần phê duyệt rõ của người dùng sau khi xem bundle này.
