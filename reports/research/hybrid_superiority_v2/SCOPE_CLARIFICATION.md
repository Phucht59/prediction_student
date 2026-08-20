# Làm rõ phạm vi metric so với đề cương

Đề cương gốc (`07 - DE-CUONG-KLTN-PTKQHT.pdf`) liệt kê Accuracy, F1-score, **R²**, **RMSE**, và Precision–Recall.

Bài toán hiện tại đã khóa thành **phân loại nhị phân nguy cơ**:

- UCI: `G3 < 10`
- OULAD: `final_result ∈ {Fail, Withdrawn}`

R² và RMSE đo hồi quy giá trị liên tục. Chúng **không phù hợp** với nhãn nhị phân. Protocol này **không** tính R²/RMSE giả tạo từ nhãn 0/1.

Thay thế khoa học:

| Vai trò | Metric |
|---|---|
| Primary ranking | Average Precision (AP) |
| Discrimination phụ | ROC-AUC |
| Ngưỡng vận hành | Risk Precision/Recall/F1, balanced accuracy |
| Ngân sách cảnh báo | Recall@10%, Recall@20% |
| Xác suất | log-loss, Brier, ECE |

Accuracy **không** đủ để tuyên bố mô hình tốt khi dữ liệu mất cân bằng (Saito & Rehmsmeier, PLOS ONE 2015).

Đề cương gốc **không bị sửa**. Đây là tài liệu làm rõ phạm vi, không phải đính chính im lặng.
