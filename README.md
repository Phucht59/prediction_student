# Xây dựng mô hình học kết hợp để dự đoán thành tích học tập sinh viên

## Mô tả ngắn

Dự án xây dựng pipeline dự đoán thành tích học tập sinh viên theo ba mức `Low`, `Medium`, `High` bằng mô hình Deep Learning CNN-BiLSTM. Sau dự đoán, module RA-HLPR tạo lộ trình học tập 4 tuần dựa trên xác suất dự đoán và chẩn đoán rủi ro học tập.

Baseline machine learning chỉ được dùng để đối chứng. Mô hình chính của đề tài là CNN-BiLSTM và module khuyến nghị downstream.

## Dataset

| Dataset | Vai trò |
|---|---|
| `student-mat` | Student Performance môn Toán |
| `student-por` | Student Performance môn Tiếng Bồ Đào Nha |
| `xAPI` | Dữ liệu hành vi học tập và tương tác học trực tuyến |

Không dùng `student-combine` làm dataset chính.

## Final Prediction Results

| Dataset | Scenario | Model | Prediction mode | Macro F1 | Recall Low | F1 Low |
|---|---|---|---|---:|---:|---:|
| student-mat | late | sequence_cnn_bilstm_only | low_f1_tuned | 0.9365 | 0.9615 | 0.8929 |
| student-por | late | sequence_cnn_bilstm_only | low_f1_tuned | 0.8783 | 0.9000 | 0.8182 |
| student-por | midterm | sequence_cnn_bilstm_only | argmax | 0.8228 | 0.6500 | 0.7429 |
| xAPI | default | gated_fusion_v28 | low_f1_tuned | 0.7541 | 0.8846 | 0.8214 |

## Final Recommender Summary

RA-HLPR là module downstream của CNN-BiLSTM:

```text
CNN-BiLSTM probabilities -> risk diagnosis -> intervention ranking -> 4-week learning path
```

- Recommender đã refresh cho `xapi` và `student-por`.
- Student-Mat recommender đang pending vì thiếu metadata checkpoint: `models/saved/final/student-mat_3class_ensemble_features.json`.
- Recommender không phải collaborative filtering.
- Không claim causal improvement vì chưa có dữ liệu phản hồi thực tế sau khi sinh viên nhận khuyến nghị.

## Cách chạy test

```powershell
py -3.10 -m pytest -q
```

Hoặc trong môi trường Python đã cài dependency:

```powershell
python -m pytest -q
```

## Cách chạy recommender

```powershell
py -3.10 scripts\run_recommender_pipeline.py --dataset xapi
py -3.10 scripts\run_recommender_pipeline.py --dataset student-por
```

Student-Mat recommender cần metadata checkpoint trước khi refresh full run.

## Báo cáo final

- `reports/final/final_model_manifest.json`
- `reports/final/final_deep_results_table.csv`
- `reports/final/final_baseline_comparison.csv`
- `reports/final/final_prediction_model_report.md`
- `reports/final/final_thesis_ready_summary.md`
- `reports/final/final_recommender_report.md`
- `reports/final/final_recommender_thesis_summary_vi.md`
- `reports/final/recommender_model_design.md`
- `reports/final/FINAL_PROJECT_STATUS.md`

## Guardrails

- Locked test chỉ dùng cho final evaluation.
- Baseline chỉ dùng đối chứng, không phải mô hình chính.
- Không dùng `student-combine`.
- Không dùng direct ADASYN với categorical label encoding.
- Không claim regression head.
- Không dùng ML baseline làm teacher hoặc distillation cho deep model/recommender.
- Không dùng true `G3`/`Class` để sinh operational recommendation.
