# Thuật ngữ mô hình dùng trong khóa luận

## Tên hiển thị

**G2 deterministic rule.** Quy tắc tham chiếu chia G2 theo đúng các vùng Low/Medium/High. Quy tắc không có xác suất hoặc uncertainty hợp lệ.

**Logistic Regression.** Mô hình tuyến tính xác suất, dùng làm baseline dễ giải thích và chi phí thấp.

**Random Forest.** Tập hợp nhiều cây quyết định, phù hợp dữ liệu bảng và quan hệ phi tuyến.

**SVM.** Mô hình biên phân tách; kernel RBF cho phép học ranh giới phi tuyến.

**HistGradientBoosting.** Boosting trên cây với histogram, dùng làm baseline cho feature tổng hợp.

**MLP.** Mạng nơ-ron truyền thẳng trên vector feature tổng hợp; là control để tách giá trị của Deep Learning khỏi thứ tự temporal.

**CNN.** Mạng tích chập một chiều, trích xuất mẫu cục bộ trong chuỗi điểm hoặc hoạt động theo tuần.

**BiLSTM.** LSTM hai chiều trên phần chuỗi đã quan sát; không được đọc sự kiện sau prediction cutoff.

**CNN–BiLSTM.** Kiến trúc kết hợp CNN và BiLSTM: CNN nhận diện mẫu cục bộ, BiLSTM tổng hợp quan hệ theo thứ tự.

**Ordinal CNN–BiLSTM.** CNN–BiLSTM dùng ordered head cho target có thứ tự Low/Medium/High. Kết quả hiện tại không chứng minh ordinal learning tốt hơn nominal learning.

**CNN–BiLSTM Ensemble.** Trung bình xác suất của ba lần huấn luyện OULAD CNN–BiLSTM với ba seed cố định. Đây là cách tổng hợp kết quả, không phải kiến trúc mạng mới.

## Ánh xạ kỹ thuật cho người phát triển

Các ID dưới đây chỉ phục vụ truy vết source/database/artifact. Phần trình bày khóa luận dùng tên mô hình ở cột phải.

| Study | Technical ID | Report name |
| --- | --- | --- |
| `student-mat` | `R0` | G2 deterministic rule |
| `student-mat` | `M1` | Random Forest |
| `student-mat` | `M2` | SVM |
| `student-mat` | `N0` | CNN–BiLSTM |
| `student-mat` | `N1` | Ordinal CNN–BiLSTM |
| `student-por` | `B-R0` | G2 deterministic rule |
| `student-por` | `B-L0` | Logistic Regression |
| `student-por` | `B-RF0` | Random Forest |
| `student-por` | `B-S0` | SVM |
| `student-por` | `B-H0` | HistGradientBoosting |
| `student-por` | `B-M0` | MLP |
| `student-por` | `B-C0` | CNN |
| `student-por` | `B-L1` | BiLSTM |
| `student-por` | `B-H1` | CNN–BiLSTM |
| `student-por` | `B-O0` | Ordinal CNN–BiLSTM |
| OULAD | `V3-MLF` | Logistic Regression |
| OULAD | `V3-MLD` | Machine Learning with Dynamic Features |
| OULAD | `V3-A0F-ENS` | MLP |
| OULAD | `V3-P0-ENS` | CNN–BiLSTM |
| OULAD | `V3-D0-ENS` | CNN–BiLSTM Ensemble |

Nguồn mapping máy đọc: [`configs/model_display_names.yaml`](../configs/model_display_names.yaml).
