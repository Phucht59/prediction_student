# KLTN — Dự đoán thành tích học tập bằng CNN–BiLSTM

Pipeline nghiên cứu cho `student-mat` của UCI Student Performance. Bài toán
phân loại ba mức từ `G3`: Low (`<= 9`), Medium (`10–14`) và High (`>= 15`).

## Phạm vi và giới hạn

- `G1` và `G2` là hai mốc điểm trước điểm cuối kỳ; chúng không phải chuỗi thời
  gian dài hạn. Mô hình CNN–BiLSTM chỉ là một kiến trúc thực nghiệm trên chuỗi
  dài hai bước.
- Kịch bản `late_stage` dùng `G1,G2`; `early_warning` loại `G2`; kịch bản
  `pre_assessment` loại cả `G1,G2`. Không được so sánh trực tiếp các kịch bản
  như có cùng lượng thông tin.
- Khuyến nghị là policy rule-based hỗ trợ cố vấn, không phải mô hình học máy và
  không được dùng để ra quyết định tự động hay kết luận quan hệ nhân quả.
- Không chỉnh sửa hoặc sử dụng số liệu trong DOCX như artifact thực nghiệm.

## Final DB-first evidence

Final run: `a2945d79-9845-4979-b148-159f4853eca3` (`completed`), được chọn từ
full nested CV 5 outer × 3 inner folds × 30 trial và chạy bằng config frozen.
Selected config: `artifacts/model_selection/nested-full-20260710/selected_config.json`
(SHA-256 `cda38460197627ac1d71e764f61d784e4c03cf6f86775339d38787c6890678ad`).
Clean-commit verification run `c719439e-bb88-42ff-bb98-d258c21d204e` reproduced
the final prediction CSV byte-for-byte; see
`artifacts/final/final-a2945d79-9845-4979-b148-159f4853eca3/reproducibility_manifest.json`.
Evidence run mới nhất nằm trong `artifacts/final/` (tên run được lưu trong
`artifacts/final/LATEST_RUN.txt`). Tất cả metric đều tính từ CSV prediction đã
lưu, không chép tay vào README.

| Phương pháp late-stage | OOF Macro-F1 | Locked-test Macro-F1 |
| --- | ---: | ---: |
| G2 threshold rule | 0.8988 | 0.9365 |
| HistGradientBoosting, toàn feature | 0.8969 | 0.9463 |
| CNN–BiLSTM, 1 seed | 0.8422 | 0.9098 |
| CNN–BiLSTM final frozen single seed | 0.8781 nested outer CV | 0.9262 |

Kết luận bắt buộc: mô hình sâu hiện chưa chứng minh giá trị tăng thêm so với
quy tắc G2 hoặc baseline đơn giản. HistGradientBoosting có điểm test cao hơn
nhưng không được chọn theo test vì OOF Macro-F1 thấp hơn G2 rule.
So sánh HGB `0.8969` và nested HGB `0.8690` dùng protocol/feature pipeline
khác nhau; xem [MODEL_COMPARISON_PROTOCOL.md](docs/MODEL_COMPARISON_PROTOCOL.md).

## Quy trình tái lập

Chạy test:

```powershell
py -3.10 -m pytest -q
```

Tạo baseline, kịch bản early-warning/pre-assessment, calibration, fairness
slice và evidence bundle:

```powershell
py -3.10 scripts/run_final_evidence.py
```

Chạy CNN-only, BiLSTM-only, CNN–BiLSTM, ablation mất cân bằng và ensemble:

```powershell
py -3.10 scripts/run_deep_ablation.py
```

Mỗi run sinh `run_manifest.json`, split hashes, OOF/locked predictions,
metrics, PR data, calibration/reliability data, fairness slices và evaluation
khuyến nghị trong `artifacts/final/<run_id>/`.

## Final DB-first pipeline

Database mặc định là `student_predict`; credentials phải lấy từ biến môi
trường, không hard-code mật khẩu. Schema lineage gồm:

```text
source_dataset_versions
source_records
ml_experiment_runs
ml_run_record_splits
ml_predictions
ml_run_metrics
ml_recommendations
```

Final pipeline yêu cầu một `selected_config.json` đã đóng băng trước khi mở
locked test. Nó không tự chạy Optuna trong lệnh final:

```powershell
py -3.10 scripts/run_pipeline.py --dataset student-mat --target-mode 3class `
  --dataset-version-id 1 --selection-config-json <duong_dan_selected_config.json>
```

`--debug` chỉ dùng smoke test, không phải kết quả khóa luận. Full model
selection dùng:

```powershell
py -3.10 scripts/optimize_model_selection.py --dataset student-mat --dataset-version-id 1 --n-trials 30 --outer-folds 5 --inner-folds 3 --selection-seed 42 --selection-run-id nested-full-20260710
```

## Guardrails

- Không đưa `G3`, `G3_raw`, nhãn thật hoặc metadata lineage vào feature/model
  input hoặc recommendation.
- Scaler, encoding, feature selection và SMOTE chỉ fit ở train portion của mỗi
  fold; locked test không đi vào Optuna, threshold hay calibration fitting.
- CNN kernel cho chuỗi hai bước là `1`; số tham số và ablation phải được ghi
  trong artifact, không diễn giải như mô hình chuỗi dài hạn.
- Class weight và SMOTE là hai lựa chọn độc lập cần ablation; không mặc định
  dùng đồng thời.
- Recommendation loại các biến `sex`, `school`, `address`, `guardian`,
  `paid`, `Dalc`, `Walc`, `goout` khỏi luật khuyến nghị tự động và luôn yêu cầu
  giảng viên/cố vấn duyệt.
