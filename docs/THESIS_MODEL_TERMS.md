# Thuật ngữ mô hình dùng trong khóa luận

## Tên hiển thị

**Logistic Regression.** Mô hình tuyến tính xác suất, dùng làm baseline dễ giải thích và chi phí thấp.

**Random Forest.** Tập hợp nhiều cây quyết định, phù hợp với dữ liệu bảng và quan hệ phi tuyến.

**SVM.** Mô hình biên phân tách; kernel RBF cho phép học ranh giới phi tuyến.

**HistGradientBoosting.** Mô hình boosting trên cây với histogram, dùng làm baseline mạnh cho feature tổng hợp.

**MLP.** Mạng nơ-ron truyền thẳng trên vector feature tổng hợp, dùng để kiểm tra giá trị của Deep Learning khi không khai thác thứ tự chuỗi.

**CNN.** Mạng tích chập một chiều, trích xuất các mẫu cục bộ trong chuỗi hoạt động học theo tuần.

**BiLSTM.** Mạng LSTM hai chiều, tổng hợp quan hệ trong phần chuỗi đã quan sát mà không đọc dữ liệu sau prediction cutoff.

**CNN–BiLSTM.** Kiến trúc kết hợp CNN và BiLSTM: CNN nhận diện mẫu cục bộ, BiLSTM học quan hệ theo chuỗi.

**CNN–BiLSTM Ensemble.** Trung bình xác suất của ba lần huấn luyện CNN–BiLSTM với ba seed cố định. Đây là cách tổng hợp kết quả, không phải một kiến trúc mạng mới.

## Bảng ánh xạ kỹ thuật cho người phát triển

Bảng này chỉ phục vụ truy vết source/database/artifact, không dùng làm tên mô hình trong phần trình bày khóa luận.

| Technical ID | Report name |
| --- | --- |
| `V3-MLF` | Logistic Regression |
| `V3-MLD` | Machine Learning with Dynamic Features |
| `V3-A0F-ENS` | MLP |
| `V3-A1-ENS` | MLP |
| `V3-H2TF-ENS` | CNN–BiLSTM |
| `V3-H3CF-ENS` | CNN–BiLSTM |
| `V3-P0-ENS` | CNN–BiLSTM |
| `V3-D0-ENS` | CNN–BiLSTM Ensemble |

Nguồn ánh xạ máy đọc là [`configs/model_display_names.yaml`](../configs/model_display_names.yaml).
