# 01. Verified Results Tables

Tài liệu này chỉ tổng hợp các bảng có thể đưa vào báo cáo khóa luận sau khi đối chiếu source code, artifact final, checkpoint metadata và output thực nghiệm hiện có. Các ô không đủ chứng cứ được ghi rõ `MISSING`, `INFERRED` hoặc `PARTIALLY VERIFIED`.

## 1. Bảng Mô Tả Datasets

| Dataset | Nguồn | Số mẫu | Nhãn | Scenario | File xử lý | Trạng thái | Evidence path |
|---|---:|---:|---|---|---|---|---|
| student-mat | UCI Student Performance, file cấu hình `student-mat.csv`; raw file không có trong workspace hiện tại | 395 `INFERRED` từ checkpoint recommender train 316 + stale locked output 79; raw không kiểm trực tiếp | `G3` -> 3 lớp bằng bins `[0,9,14,20]` | `late` trong final status | `src/config.py`, `src/data_pipeline.py`, `archive/experiments/src_experiments/current_src_experiments.20260617_170850/common.py` | `PARTIALLY VERIFIED`: nhãn và scenario có code; raw/split final thiếu | `src/config.py`, `src/data_pipeline.py`, `reports/final/FINAL_PROJECT_STATUS.md`, `models/recommendation/student-mat_mlp.pt` |
| student-por | UCI Student Performance, file cấu hình `student-por.csv`; raw file không có trong workspace hiện tại | 649 `VERIFIED/INFERRED` từ checkpoint recommender train 519 + locked output 130 | `G3` -> 3 lớp bằng bins `[0,9,14,20]` | `late`, `midterm` trong final status | `src/config.py`, `src/data_pipeline.py`, archive scenario pipeline | `PARTIALLY VERIFIED`: kết quả prediction có trong final status; recommender refreshed có output; raw/split final thiếu | `src/config.py`, `src/data_pipeline.py`, `reports/final/FINAL_PROJECT_STATUS.md`, `outputs/recommender/student-por/recommender_metrics.json` |
| xAPI | xAPI Educational Mining, file cấu hình `xAPI-Edu-Data.csv`; raw file không có trong workspace hiện tại | 480 `VERIFIED/INFERRED` từ checkpoint recommender train 384 + locked output 96 | `Class` với mapping `L=0`, `M=1`, `H=2` | `default` trong final status | `src/config.py`, `src/data_pipeline.py`, `scripts/run_pipeline.py` | `VERIFIED` cho final prediction và recommender offline output; raw file vẫn thiếu trong workspace | `reports/final/final_model_manifest.json`, `reports/final/final_deep_results_table.csv`, `outputs/recommender/xapi/recommender_metrics.json` |

## 2. Bảng Feature Theo Scenario

| Dataset/Scenario | Sequence features | Context/input features | Feature bị loại để tránh leakage | Trạng thái | Evidence path |
|---|---|---|---|---|---|
| student `early` | Không dùng sequence | Toàn bộ feature hợp lệ sau khi loại `G1`, `G2`, target và `G3_raw`; feature engineering chạy sau khi drop grade chưa khả dụng | `G3`, `G3_raw`, `G1`, `G2` | `VERIFIED` trong archive scenario code; không phải final row | `archive/experiments/src_experiments/current_src_experiments.20260617_170850/common.py` |
| student `midterm` | `G1` | Feature hợp lệ sau khi loại `G2`, target và `G3_raw`; không được sinh feature dẫn xuất từ `G2` | `G3`, `G3_raw`, `G2` | `VERIFIED` trong archive scenario code; final row student-por midterm `PARTIALLY VERIFIED` | `archive/experiments/src_experiments/current_src_experiments.20260617_170850/common.py`, `reports/final/FINAL_PROJECT_STATUS.md` |
| student `late` | `G1`, `G2` | Feature hợp lệ ngoài target; `G1`, `G2` cũng là chuỗi học tập | `G3`, `G3_raw` khỏi input context | `VERIFIED` trong source; final rows `PARTIALLY VERIFIED` | `src/data_pipeline.py`, `archive/experiments/src_experiments/current_src_experiments.20260617_170850/common.py`, `reports/final/FINAL_PROJECT_STATUS.md` |
| xAPI `default` | `raisedhands`, `VisITedResources`, `AnnouncementsView`, `Discussion` | Các biến categorical/numerical còn lại sau khi loại `Class`; categorical được label-encode, numerical MinMax scale | `Class` | `VERIFIED` | `src/data_pipeline.py`, `reports/final/final_model_manifest.json` |

## 3. Bảng Cấu Hình CNN-BiLSTM

| Model/source | Input tensor shape | CNN | BiLSTM | Fusion/context | Dropout | Loss/optimizer | Trạng thái | Evidence path |
|---|---|---|---|---|---:|---|---|---|
| `StudentHybridModel` current | `seq_x`: `(batch, seq_len, 1)`; `num_x`, `cat_x` context tensors | `Conv1d(1, cnn_channels, kernel_size, padding=kernel_size//2)` + BatchNorm + ReLU + Dropout | Bidirectional LSTM với `hidden_size=bilstm_hidden` | Attention pooling sequence + context MLP + concat fusion classifier | config default `0.3` | CrossEntropy/Focal/ClassBalanced tùy config; Adam/AdamW trong train pipeline | `VERIFIED` source architecture; không chứng minh là exact final checkpoint | `src/models/models.py`, `src/train_pipeline.py`, `src/config.py` |
| `SequenceCNNBiLSTMOnly` archive | Student sequence only, thường `(batch, 1 hoặc 2, 1)` theo scenario | Conv1d + BatchNorm + ReLU + Dropout; default `cnn_channels=32`, `kernel_size=3` | Bidirectional LSTM default `hidden_dim=64` | Attention pooling; class/ordinal/reg heads trong code thử nghiệm | default `0.15` | Training/eval archive; threshold OOF | `PARTIALLY VERIFIED`: tên model khớp final Student rows, nhưng final checkpoint/manifest thiếu | `archive/experiments/deep_debug.py`, `reports/final/FINAL_PROJECT_STATUS.md` |
| `StudentHybridV27` / gated source | `seq_x`, `num_x`, `cat_x` | Conv1d + BatchNorm + ReLU + Dropout | Bidirectional LSTM | `GatedFusion`: gate giữa vector sequence và context | default `0.3` | `JointHybridLoss` hỗ trợ class/ordinal/reg trong source | `VERIFIED` source gated fusion; exact `gated_fusion_v28` source/checkpoint `MISSING` | `src/models_v27.py`, `src/losses_v27.py`, `reports/final/final_model_manifest.json` |

## 4. Bảng Cấu Hình Gated Fusion

| Thành phần | Mô tả đã xác minh | Final claim được phép | Trạng thái | Evidence path |
|---|---|---|---|---|
| `GatedFusion` | Tạo `h_seq = proj_seq(seq_vec)`, `h_ctx = proj_ctx(ctx_vec)`, `gate = sigmoid(gate([seq_vec, ctx_vec]))`, sau đó `fused = gate*h_seq + (1-gate)*h_ctx` | Có thể mô tả là cơ chế học trọng số động giữa thông tin chuỗi và ngữ cảnh | `VERIFIED` source | `src/models_v27.py` |
| `gated_fusion_v28` final xAPI | Artifact final ghi model variant `gated_fusion_v28`, architecture `CNN-BiLSTM with gated context fusion`, prediction mode `low_f1_tuned` | Có thể claim là final xAPI model theo artifact final | `VERIFIED` result, `MISSING` exact source class/checkpoint mang tên v28 | `reports/final/final_model_manifest.json`, `reports/final/final_deep_results_table.csv` |
| Regression head | Có trong source v27/archive nhưng final report loại trừ claim regression head | Không claim mô hình final là multi-task regression hoặc có regression head vận hành | `VERIFIED guardrail` | `reports/final/final_model_manifest.json`, `reports/final/final_prediction_model_report.md` |

## 5. Bảng Baseline

| Dataset | Baseline model | Prediction mode | Macro F1 | Low metrics | Vai trò | Trạng thái | Evidence path |
|---|---|---|---:|---|---|---|---|
| xAPI | `RandomForestClassifier` | `argmax` | 0.8465 | `not_available` trong final CSV | Đối chứng, không phải mô hình chính, không dùng teacher/distillation/feature importance | `VERIFIED` | `reports/final/final_baseline_comparison.csv`, `reports/final/final_prediction_model_report.md` |
| student-mat | Logistic Regression, Random Forest, XGBoost/CatBoost/HistGradient fallback, MLP trong archive | Không xác định final | `MISSING` final baseline artifact | `MISSING` | Đối chứng thử nghiệm | `INFERRED/PARTIAL` source only | `archive/experiments/baselines.py` |
| student-por | Logistic Regression, Random Forest, XGBoost/CatBoost/HistGradient fallback, MLP trong archive | Không xác định final | `MISSING` final baseline artifact | `MISSING` | Đối chứng thử nghiệm | `INFERRED/PARTIAL` source only | `archive/experiments/baselines.py` |

## 6. Bảng Kết Quả Final Deep Models

| Dataset | Scenario | Final model | Prediction mode | Macro F1 | Recall Low | F1 Low | Trạng thái kiểm chứng | Evidence path |
|---|---|---|---|---:|---:|---:|---|---|
| student-mat | late | `sequence_cnn_bilstm_only` | `low_f1_tuned` | 0.9365 | 0.9615 | 0.8929 | `Partially verified`: có trong final status/README; thiếu manifest/checkpoint/per-run CSV final tương ứng | `reports/final/FINAL_PROJECT_STATUS.md`, `README.md`, `CLEANUP_LOG.md` |
| student-por | late | `sequence_cnn_bilstm_only` | `low_f1_tuned` | 0.8783 | 0.9000 | 0.8182 | `Partially verified`: có trong final status/README; thiếu manifest/checkpoint/per-run CSV final tương ứng | `reports/final/FINAL_PROJECT_STATUS.md`, `README.md`, `CLEANUP_LOG.md` |
| student-por | midterm | `sequence_cnn_bilstm_only` | `argmax` | 0.8228 | 0.6500 | 0.7429 | `Partially verified`: có trong final status/README; thiếu manifest/checkpoint/per-run CSV final tương ứng | `reports/final/FINAL_PROJECT_STATUS.md`, `README.md`, `CLEANUP_LOG.md` |
| xAPI | default | `gated_fusion_v28` | `low_f1_tuned` | 0.7541 | 0.8846 | 0.8214 | `Verified`: có final manifest, final deep table, final status | `reports/final/final_model_manifest.json`, `reports/final/final_deep_results_table.csv`, `reports/final/FINAL_PROJECT_STATUS.md` |

## 7. Bảng So Sánh Deep Model Với Baseline

| Dataset | Deep model Macro F1 | Baseline tốt nhất trong final artifact | Baseline Macro F1 | Kết luận hợp lệ | Trạng thái | Evidence path |
|---|---:|---|---:|---|---|---|
| xAPI | 0.7541 | Random Forest | 0.8465 | Baseline RF cao hơn deep model về Macro F1; deep model được chọn vì ưu tiên nhận diện Low và tích hợp probability cho RA-HLPR | `VERIFIED` | `reports/final/final_baseline_comparison.csv`, `reports/final/final_prediction_model_report.md` |
| student-mat | 0.9365 | `MISSING` final baseline artifact | `MISSING` | Không được claim deep model tốt hơn baseline nếu không có artifact | `MISSING/PARTIAL` | `reports/final/FINAL_PROJECT_STATUS.md`, `archive/experiments/baselines.py` |
| student-por late | 0.8783 | `MISSING` final baseline artifact | `MISSING` | Không được claim deep model tốt hơn baseline nếu không có artifact | `MISSING/PARTIAL` | `reports/final/FINAL_PROJECT_STATUS.md`, `archive/experiments/baselines.py` |
| student-por midterm | 0.8228 | `MISSING` final baseline artifact | `MISSING` | Không được claim deep model tốt hơn baseline nếu không có artifact | `MISSING/PARTIAL` | `reports/final/FINAL_PROJECT_STATUS.md`, `archive/experiments/baselines.py` |

## 8. Bảng Kết Quả Nhận Diện Lớp Low

| Dataset | Scenario | Prediction mode | Recall Low | F1 Low | Ý nghĩa diễn giải | Trạng thái | Evidence path |
|---|---|---|---:|---:|---|---|---|
| student-mat | late | `low_f1_tuned` | 0.9615 | 0.8929 | Ưu tiên phát hiện sinh viên nguy cơ thấp trong final status | `Partially verified` | `reports/final/FINAL_PROJECT_STATUS.md` |
| student-por | late | `low_f1_tuned` | 0.9000 | 0.8182 | Ưu tiên phát hiện Low ở scenario muộn | `Partially verified` | `reports/final/FINAL_PROJECT_STATUS.md` |
| student-por | midterm | `argmax` | 0.6500 | 0.7429 | Không dùng threshold tuning trong final row midterm | `Partially verified` | `reports/final/FINAL_PROJECT_STATUS.md` |
| xAPI | default | `low_f1_tuned` | 0.8846 | 0.8214 | Có final manifest và deep table xác nhận | `Verified` | `reports/final/final_model_manifest.json`, `reports/final/final_deep_results_table.csv` |

## 9. Bảng Trạng Thái Recommender Theo Dataset

| Dataset | Trạng thái recommender | Số hồ sơ locked output | Metric offline chính | Checkpoint risk head | Được phép claim | Evidence path |
|---|---|---:|---|---|---|---|
| xAPI | Refreshed final output có metrics, paths và recommendations | 96 | Risk F1 macro 0.9831; P@3 0.6840; NDCG@3 0.8229; path risk coverage 0.8958 | `models/recommendation/xapi_mlp.pt` | Có thể claim offline risk/path generation; không claim cải thiện điểm thật | `outputs/recommender/xapi/recommender_metrics.json`, `models/recommendation/xapi_mlp.pt`, `reports/final/final_recommender_report.md` |
| student-por | Refreshed final output có metrics, paths và recommendations | 130 | Risk F1 macro 0.9359; P@3 0.6641; NDCG@3 0.7455; path risk coverage 0.9508 | `models/recommendation/student-por_mlp.pt` | Có thể claim offline risk/path generation; không claim cải thiện điểm thật | `outputs/recommender/student-por/recommender_metrics.json`, `models/recommendation/student-por_mlp.pt`, `reports/final/final_recommender_report.md` |
| student-mat | Pending final refreshed recommender; stale archived output tồn tại nhưng không dùng làm final | 79 stale archive only | Không dùng làm final | `models/recommendation/student-mat_mlp.pt` tồn tại | Chỉ nói pending/missing prediction metadata | `reports/final/final_recommender_report.md`, `outputs/recommender/archive/stale_student_mat_pending_prediction_metadata/` |

## 10. Bảng Limitations

| Limitation | Ảnh hưởng | Trạng thái | Evidence path |
|---|---|---|---|
| Raw datasets và processed splits không có trong tracked workspace | Không kiểm trực tiếp missing values, duplicates, exact split rows bằng raw CSV | `MISSING` | `data/raw/`, `data/processed/`, `.gitignore` |
| `models/saved/final` trống | Không thể rerun full final training/recommender từ script hiện tại | `MISSING` | `scripts/run_pipeline.py`, `scripts/run_recommender_pipeline.py`, `models/saved/final/` |
| Student final rows thiếu manifest/checkpoint exact | Student metrics chỉ partially verified | `PARTIALLY VERIFIED` | `reports/final/FINAL_PROJECT_STATUS.md`, `models/final/final_model_manifest.json` |
| `models/final/final_model_manifest.json` mâu thuẫn với final reports | Không dùng file này làm nguồn final metrics | `CONTRADICTED/NON-FINAL` | `models/final/final_model_manifest.json`, `reports/final/final_model_manifest.json` |
| Không có statistical significance test | Không được claim khác biệt có ý nghĩa thống kê | `MISSING` | Search repo không có final statistical test artifact |
| Không có user feedback hoặc A/B test recommender | Không được claim RA-HLPR cải thiện thành tích thực tế | `MISSING` | `reports/final/final_recommender_report.md`, recommender outputs |
| Baseline final đầy đủ chỉ có xAPI | Không claim deep tốt hơn baseline trên Student datasets | `MISSING/PARTIAL` | `reports/final/final_baseline_comparison.csv`, `archive/experiments/baselines.py` |
| Exact numeric threshold cho Student final rows thiếu artifact | Không ghi threshold cụ thể nếu chưa tìm được | `MISSING/PARTIAL` | `archive/experiments/deep_debug.py`, `reports/final/FINAL_PROJECT_STATUS.md` |
