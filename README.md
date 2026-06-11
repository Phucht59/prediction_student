# Hệ thống Dự đoán và Khuyến nghị Thành tích Học tập (V27 SOTA Tabular-Attention)

Dự án này là mã nguồn phục vụ cho **Khóa luận Tốt nghiệp**, áp dụng các kiến trúc State-of-The-Art (SOTA) từ lĩnh vực Deep Learning Tabular (DeepFM, DCN-V2, FT-Transformer) kết hợp với Trí tuệ Nhân tạo Có thể Giải thích (Explainable AI) và Hệ thống Sinh Khuyến nghị Phản thực tế (Counterfactual Recommendation).

Mục tiêu chính: Dự đoán thành tích của học sinh (G3/Class) và đề xuất lộ trình hành động thiết thực giúp học sinh nâng cao điểm số.

## Kiến trúc Mã nguồn (Siêu tinh gọn)

Mã nguồn đã được thiết kế lại theo hướng **phẳng và cực kỳ tối giản**, chỉ giữ lại những luồng logic tối quan trọng nhất để dễ đọc, dễ chạy và dễ đưa vào báo cáo KLTN.

```text
├── data/raw/                 # Chứa dữ liệu gốc (student-mat.csv, student-por.csv, xAPI-Edu-Data.csv)
├── reports/v27/              # Báo cáo đầu ra (metrics, hình ảnh, khuyến nghị, v.v.)
├── scripts/
│   └── run_pipeline.py       # Script thực thi luồng E2E duy nhất
└── src/                      # Mã nguồn Lõi (Siêu tinh gọn)
    ├── config.py             # File cấu hình tham số, seed, và hyperparams
    ├── utils.py              # Các hàm tiện ích (Logging, Seed fixing)
    ├── data_pipeline.py      # Tiền xử lý, Feature Engineering, Feature Selection và Dataloader
    ├── models.py             # Kiến trúc V27 (DeepFM, DCN-V2, FT-Transformer) và Factory
    ├── train_pipeline.py     # Huấn luyện mô hình (Optuna Search & 5-Seed Ensemble)
    ├── evaluation.py         # Chạy đánh giá tập Locked Test (đảm bảo không leakage)
    └── explainability.py     # XAI (Permutation Importance) & Sinh Text Khuyến nghị
```

## Các Công nghệ & Kiến trúc Lõi
- **Mô hình**: `DeepFM`, `DCN-V2` (Deep & Cross Network), `FT-Transformer` (Self-Attention trên Tabular Data).
- **AutoML**: Sử dụng `Optuna` dò tìm 50 trials tự động trên 5-Folds Cross Validation.
- **Robustness**: Kỹ thuật `5-Seed Ensemble` để loại bỏ nhiễu ngẫu nhiên. Chống mất cân bằng dữ liệu bằng `ADASYN/SMOTE` + `Focal Loss`.
- **Đánh giá chặt chẽ**: `Locked Test 20%` được tách hoàn toàn biệt lập trước mọi quy trình xử lý dữ liệu.
- **Explainable AI (XAI)**:
  - Phân tích Feature Importance thông qua `Permutation Importance`.
  - Thuật toán `Greedy Counterfactual Search` tìm kiếm giải pháp tối ưu thay đổi các thuộc tính hành vi (actionables) để dự báo tăng hạng.

## Hướng dẫn Chạy (Quickstart)

### 1. Cài đặt Môi trường
Cài đặt toàn bộ các thư viện cần thiết thông qua requirements:
```powershell
pip install -r requirements.txt
```

### 2. Chuẩn bị Dữ liệu
Đảm bảo 3 file dữ liệu gốc đang tồn tại trong thư mục `data/raw/`:
- `student-mat.csv`
- `student-por.csv`
- `xAPI-Edu-Data.csv`

### 3. Thực thi Pipeline End-to-End
Sử dụng script E2E duy nhất cho từng dataset (chọn `student-mat`, `student-por` hoặc `xapi`). Hệ thống sẽ tự động chạy quy trình: Data Split -> Optuna Search -> Train Ensemble -> Evaluate -> Explainability.

```powershell
python scripts/run_pipeline.py --dataset student-mat --target-mode 3class
```
*(Thêm cờ `--debug` nếu muốn chạy test cực nhanh với 2 trials Optuna và 3 epochs).*

### 4. Kết quả (Output)
Toàn bộ kết quả tự động xuất ra thư mục `reports/v27/` bao gồm:
- **`metrics/`**: Json file chứa F1-Macro, Accuracy, RMSE...
- **`predictions/`**: CSV chứa kết quả dự đoán của mô hình Ensemble.
- **`explanations/`**: Phân tích tầm quan trọng của các Feature.
- **`recommendations/`**: Báo cáo sinh Text Khuyến nghị chi tiết cho từng sinh viên, kèm các chỉ số Validity/Sparsity.
