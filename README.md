# Dự đoán kết quả và rủi ro học tập bằng CNN-BiLSTM

Repository phát hành ba mô hình dự đoán sinh viên và một hệ thống hỗ trợ khuyến nghị dựa trên rủi ro. Bản phát hành chỉ tổng hợp bằng chứng cuối đã đóng băng; các lệnh kiểm định không huấn luyện lại mô hình.

## Mục tiêu

- Phân loại kết quả Low/Medium/High trên Student-Mat và Student-Por.
- Phát hiện sớm Not-at-risk/At-risk trên OULAD.
- Chuyển hồ sơ rủi ro thành kế hoạch hỗ trợ có kiểm soát và chờ đánh giá chuyên gia.

## Dataset và mô hình chính thức

- Dataset IDs: `student-mat`, `student-por`, `OULAD`.
- **CNN-BiLSTM — Student-Mat** (`cnn_bilstm_mat`)
- **CNN-BiLSTM — Student-Por** (`cnn_bilstm_por`)
- **CNN-BiLSTM — OULAD** (`cnn_bilstm_oulad`)
- **Student Risk-Based Recommendation System** (`student_risk_recommendation_system`)

CNN trích xuất mẫu cục bộ trong chuỗi đặc trưng, BiLSTM mô hình hóa quan hệ hai chiều, và đầu ra xác suất được tổng hợp trên các outer-fold/seed đã đăng ký. Hệ thống khuyến nghị chỉ dùng kết quả rủi ro để hỗ trợ quyết định; không tuyên bố hiệu quả can thiệp nhân quả.

## Kết quả chính

| Dataset | Mô hình | Macro-F1 | Balanced Accuracy | PR-AUC |
|---|---|---:|---:|---:|
| Student-Mat | CNN-BiLSTM — Student-Mat | 0.9015 | 0.9021 | 0.9442 |
| Student-Por | CNN-BiLSTM — Student-Por | 0.8623 | 0.8676 | 0.9147 |
| OULAD | CNN-BiLSTM — OULAD | 0.8281 | 0.8203 | 0.8934 |

Bảng đủ chín mô hình, chỉ số từng lớp, Top-k, confusion matrix, nguồn và checksum nằm trong [báo cáo cuối](reports/final/FINAL_MODEL_RESULTS.md). Ô thiếu bằng chứng cuối được ghi `N/A`, không ước lượng.

## Kiến trúc repository

- `configs/final/`: cấu hình và registry công khai.
- `src/models/`, `src/evaluation/`, `src/recommendation/`: API chính thức.
- `artifacts/final/`: JSON/CSV, registry và checksum canonical.
- `reports/final/`: bảng kết quả và giới hạn tuyên bố.
- `docs/`: kiến trúc, dữ liệu, protocol và tái lập.

## Validation

```powershell
python project.py final status
python project.py final report
python project.py final validate
```

Các lệnh trên chỉ đọc/tổng hợp frozen evidence. Future OULAD được giữ khóa. Nhãn chuyên gia chưa có vẫn mang trạng thái `PENDING_EXPERT_LABELS`.

## Giới hạn khoa học

Kết quả chỉ áp dụng cho target, split, seed và dữ liệu đã đăng ký. Không suy diễn quan hệ nhân quả từ dự đoán hoặc khuyến nghị; không tuyên bố ưu thế ngoài miền khi bằng chứng đóng băng chưa xác lập. Xem [Claim Boundaries](reports/final/CLAIM_BOUNDARIES.md).
