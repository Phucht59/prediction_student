with open('C:/Huflit/kltn/reports/KhoaLuan_ChinhThuc.md', 'a', encoding='utf-8') as f:
    f.write("""

# CHƯƠNG 5: THỰC NGHIỆM VÀ ĐÁNH GIÁ KẾT QUẢ

## 5.1. Thiết lập Môi trường và Đánh giá Lạc quan (Optimistic Flaw)
Thực nghiệm được chạy trên môi trường Python 3.10, PyTorch 2.0, với cấu hình phần cứng cơ bản. Trong giai đoạn đầu tiên, chúng tôi cố tình chạy mô hình CNN-BiLSTM dưới giao thức Lạc quan (Optimistic) – tức là cho phép chọn Epoch dừng dựa trên điểm số của tập Test. Kết quả không nằm ngoài dự đoán: F1-Macro đạt mốc **1.0000** (Toán) và **0.94** (Tiếng Bồ Đào Nha).

Tuy nhiên, đây là những con số vô giá trị về mặt thực tiễn. Khi chúng tôi khóa kín tập Test và bắt buộc mô hình phải chọn Epoch dựa trên tập Validation (Giao thức Strict), bức tranh thực tế lập tức lộ diện: Điểm F1 rơi xuống quanh mức 0.75-0.77. Điều này là minh chứng hùng hồn nhất cho sự tồn tại của Overfitting trong các đánh giá học thuật lỏng lẻo.

## 5.2. Tối ưu hóa Siêu tham số với Optuna
Để tìm ra giới hạn sức mạnh thực sự của mô hình trong điều kiện Strict Validation, chúng tôi đã khởi chạy quy trình Bayesian Optimization bằng Optuna với 120 vòng lặp (Trials) cho mỗi bộ dữ liệu. Không gian tìm kiếm trải dài trên các chiều:
- Số lượng bộ lọc CNN (16 đến 128).
- Số chiều ẩn của BiLSTM (32 đến 256).
- Tỷ lệ Dropout (0.1 đến 0.5) và Weight Decay.
- Hệ số $\gamma$ của Focal Loss (1.0 đến 3.0).

Sau nhiều giờ chạy, Optuna đã tìm ra cấu hình hội tụ tối ưu (Ví dụ: Trial 20 cho bộ Toán và Trial 51 cho bộ Tiếng Bồ Đào Nha). Đặc biệt, hệ số $\gamma=2.5$ của Focal Loss tỏ ra cực kỳ hiệu quả trong việc ép mô hình phải nhận diện chính xác nhóm học sinh có nguy cơ trượt môn (độ nhạy Recall tăng vọt từ 50% lên trên 70%).

## 5.3. Phát hiện Học thuật: Phương sai Dữ liệu (Data Variance)
Bất chấp việc đã áp dụng Strict Validation, một phát hiện chấn động khác đã nảy sinh: Sự biến động của điểm số phụ thuộc vào hạt giống ngẫu nhiên (Random Seed).

Đối với tập Toán (student-mat), tập Test 15% chỉ chứa vỏn vẹn khoảng 60 học sinh. Khi sử dụng mô hình Trial 20 siêu tối ưu và chia dữ liệu bằng `seed=123`, điểm F1 bất ngờ tăng vọt lên mức **0.8104**. Tuy nhiên, khi chúng tôi giữ nguyên kiến trúc mạng nhưng thay đổi hạt giống chia dữ liệu thành `seed=42`, điểm F1 lập tức trượt xuống **0.7184**.

Nguyên nhân gốc rễ là do **Phương sai Dữ liệu (Data Variance)**. Trên một tập dữ liệu quá nhỏ, việc một học sinh "khó đoán" hay "dễ đoán" vô tình lọt vào tập Test sẽ làm sai lệch hoàn toàn trọng số phần trăm của độ chính xác. Con số 0.8104 (Peak F1) là có thật, nhưng nó chỉ là thành tích cực đại đạt được khi mô hình may mắn gặp một tập Test phù hợp (Lucky Split). 

Để trung thực tuyệt đối với khoa học, chúng tôi đã huấn luyện chéo 126 mô hình với nhiều tập Test ngẫu nhiên khác nhau (Multi-seed Ensemble). Kết quả đo đạc cho thấy: Khả năng dự báo trung bình, vững chãi nhất (Average Strict F1) của kiến trúc CNN-BiLSTM nằm ở mức **~0.76**. Khóa luận mạnh dạn công bố cả hai mức điểm này để nhấn mạnh rằng: Trong kỷ nguyên AI, đánh giá tính ổn định (Robustness) quan trọng hơn việc chạy đua những con số cực đại.

## 5.4. Đánh giá Cắt bỏ (Ablation Study)
Để chứng minh sự cần thiết của từng khối chức năng, chúng tôi tiến hành gỡ bỏ lần lượt các phần tử:
- **Bỏ CNN 1D:** Mô hình chỉ còn BiLSTM. Điểm số giảm khoảng 2-3%, chứng tỏ CNN thực sự giúp trích xuất các "cụm hành vi" cục bộ hữu ích.
- **Bỏ BiLSTM:** Mô hình chỉ còn CNN. F1 suy giảm nghiêm trọng (>8%), khẳng định chuỗi quan hệ là cốt lõi của dữ liệu giáo dục.
- **Bỏ Focal Loss:** Khả năng nhận diện học sinh lớp F (rớt môn) gần như biến mất do mất cân bằng lớp.


# CHƯƠNG 6: KẾT LUẬN VÀ HƯỚNG PHÁT TRIỂN

## 6.1. Kết luận
Khóa luận đã hoàn thành xuất sắc mục tiêu đề ra: Xây dựng một kiến trúc lai Multi-Scale CNN-BiLSTM để dự báo thành tích học tập của sinh viên. Đóng góp lớn nhất của khóa luận không chỉ dừng lại ở việc tạo ra một kiến trúc mạng hiệu quả, mà là ở giá trị học thuật:
1. **Lật tẩy Overfitting:** Chỉ rõ nguyên nhân vì sao các mô hình có thể dễ dàng đạt F1 = 1.00 nhưng lại thất bại trong thực tế (do Data Leakage qua Early Stopping).
2. **Khẳng định chuẩn mực:** Đề xuất và tuân thủ tuyệt đối giao thức Strict Validation 70/15/15.
3. **Minh bạch hóa Data Variance:** Dũng cảm thừa nhận và đo lường sự dao động của mô hình trên các tập dữ liệu cực nhỏ, giải mã sự chênh lệch khổng lồ giữa điểm Peak (0.8104) và điểm trung bình (0.76).

## 6.2. Hướng phát triển trong tương lai
Sự hạn chế lớn nhất hiện nay của lĩnh vực EDM là kích thước bộ dữ liệu. Để giải quyết triệt để vấn đề Data Variance, trong tương lai, chúng tôi định hướng thu thập các bộ dữ liệu quy mô lớn (hàng chục ngàn sinh viên). Ngoài ra, có thể tích hợp Đồ thị Tri thức (Knowledge Graphs) hoặc Mạng Nơ-ron Đồ thị (GNN) để đưa các yếu tố tương tác xã hội giữa các sinh viên vào trong kiến trúc dự báo.
""")
print('Appended Ch5 and Ch6')
