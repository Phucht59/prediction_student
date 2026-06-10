with open('C:/Huflit/kltn/reports/KhoaLuan_ChinhThuc.md', 'a', encoding='utf-8') as f:
    f.write("""

# CHƯƠNG 3: NGHIÊN CỨU LIÊN QUAN

## 3.1. Các nghiên cứu ứng dụng EDM truyền thống
Trong những năm đầu của lĩnh vực Khai phá dữ liệu giáo dục (EDM), các kỹ thuật học máy cơ bản thường xuyên được áp dụng để giải quyết bài toán phân loại và hồi quy kết quả học tập. Một trong những công trình mang tính nền tảng là nghiên cứu của Cortez và Silva (2008) khi công bố bộ dữ liệu Student Performance. Bằng việc áp dụng Cây quyết định (Decision Tree) và Rừng ngẫu nhiên (Random Forest), các tác giả đã chứng minh mối liên hệ mật thiết giữa nền tảng gia đình, thói quen uống rượu và kết quả học tập.

Tiếp nối đó, các mô hình Máy học Vector Hỗ trợ (SVM) và K-Láng giềng gần nhất (KNN) đã được triển khai rộng rãi trên nhiều bộ dữ liệu khác nhau. Mặc dù đạt được những thành tựu nhất định, giới hạn lớn nhất của học máy truyền thống nằm ở việc phụ thuộc quá lớn vào kỹ thuật chế tác đặc trưng thủ công (Manual Feature Engineering). Khả năng khám phá các mối tương quan phi tuyến tính bậc cao giữa các thuộc tính hành vi bị giới hạn bởi cấu trúc thuật toán.

## 3.2. Sự trỗi dậy của Học sâu trong Dự báo điểm số
Với khả năng tự động học biểu diễn (Representation Learning), Học sâu (Deep Learning) dần thay thế các mô hình học máy truyền thống trong EDM. Mạng Nơ-ron Nhân tạo nhiều lớp (MLP) là nỗ lực đầu tiên để ánh xạ đặc trưng đa chiều. Tuy nhiên, sức mạnh thực sự bùng nổ khi các cấu trúc mạng phức tạp hơn như CNN và LSTM được kết hợp.

Gần đây, một số nghiên cứu đã công bố một khung kiến trúc Hybrid kết hợp CNN và BiLSTM. Thông qua việc biến đổi dữ liệu nhân khẩu học và hành vi thành các véc-tơ đầu vào cho 1D-CNN, sau đó phân tích chuỗi đặc trưng bằng BiLSTM, tác giả đã báo cáo một sự vượt trội về độ chính xác (Accuracy) và F1-score so với các mô hình baseline như Random Forest hay SVM. Tuy nhiên, sự đột phá về con số này cũng là điểm khởi đầu cho cuộc điều tra sâu sắc của khóa luận chúng tôi.

## 3.3. Lật tẩy Lỗ hổng Đánh giá (Evaluation Flaw) trong các nghiên cứu trước
Khi tái hiện lại cấu trúc Hybrid CNN-BiLSTM từ các báo cáo tiền nhiệm, chúng tôi đã đạt được những chỉ số kinh ngạc: F1-Macro đạt 1.0000 trên tập môn Toán và 0.94 trên tập môn Tiếng Bồ Đào Nha. Một mô hình hoàn hảo đến mức không tưởng. Quá trình giải mã mã nguồn (reverse-engineering) đã lật tẩy một sai lầm chết người trong giao thức đánh giá của bài báo gốc: **Sự giao thoa của tập Kiểm thử (Test set leakage)**.

Cụ thể, các nghiên cứu trước phân chia dữ liệu thành hai tập (Ví dụ: 80% Train, 20% Test). Lẽ ra, quá trình huấn luyện chỉ được giới hạn tuyệt đối trong tập Train. Thế nhưng, các tác giả đã thiết lập hàm Dừng sớm (Early Stopping) và việc lưu mô hình tốt nhất (Model Checkpointing) bằng cách liên tục tính toán sai số trên chính tập 20% Test này sau mỗi vòng lặp (Epoch). Điều này đồng nghĩa với việc mô hình đã gián tiếp "nhìn thấy" tập dữ liệu kiểm thử hàng trăm lần, và thuật toán sẽ chỉ chọn cái epoch mà mô hình tình cờ dự đoán khớp nhất với tập Test. Đây không phải là sức mạnh trí tuệ nhân tạo, đây là hành vi "khớp đường cong" (curve-fitting) ngẫu nhiên. Mốc điểm F1 cao ngất ngưởng đó là sự lạc quan giả tạo (Optimistic Flaw) và mô hình sẽ lập tức sụp đổ độ chính xác khi được triển khai cho nhóm sinh viên của học kỳ tiếp theo.


# CHƯƠNG 4: PHƯƠNG PHÁP VÀ MÔ HÌNH ĐỀ XUẤT

## 4.1. Bộ dữ liệu nghiên cứu
Khóa luận tiến hành thực nghiệm trên ba bộ dữ liệu giáo dục chuẩn mực, mang tính đại diện cao cho các bài toán EDM:
1. **Student-Mat (Môn Toán):** Bao gồm thông tin của 395 học sinh trung học. Bộ dữ liệu chứa 33 thuộc tính nhân khẩu học, lối sống và điểm số các kỳ (G1, G2, G3).
2. **Student-Por (Môn Tiếng Bồ Đào Nha):** Mở rộng từ bộ Student-Mat, bao gồm 649 học sinh học môn Tiếng Bồ Đào Nha.
3. **xAPI Educational Data:** Tập hợp hồ sơ hành vi học tập của 480 học sinh (chủ yếu là hệ thống số liệu như số lần giơ tay, số lần tham gia nhóm thảo luận, truy cập tài liệu).

Trong đề tài này, điểm số cuối kỳ (G3) của tập Student được chuyển đổi thành bài toán phân lớp 5 nhãn (5-class classification) theo khung điểm của hệ thống giáo dục Bồ Đào Nha: Khá giỏi (A, B, C), Trung bình (D) và Trượt (F). Bộ xAPI được dự đoán theo 3 nhãn: H (Cao), M (Trung bình), L (Thấp).

## 4.2. Khai phá Siêu Đặc Trưng (Ultra Feature Engineering)
Để khắc phục nhược điểm về khối lượng mẫu hữu hạn (đặc biệt là 395 học sinh của tập Toán), chúng tôi đề xuất chiến lược xử lý dữ liệu sâu (Deep/Ultra Engineered Features), chia làm ba mũi nhọn:
- **Biến đổi Cấu trúc (Structural Transformation):** Biến đổi toàn bộ các biến Categorical thành One-hot Encoding và Ordinal Encoding để mạng nơ-ron có thể hấp thụ tín hiệu toán học.
- **Đặc trưng Đa thức (Polynomial Features):** Đối với các biến liên tục quan trọng như G1, G2, sự khác biệt giữa các mức điểm được làm rõ nét bằng cách tạo ra các biến bậc 2 ($G1^2, G2^2$) và biến tương tác ($G1 \times G2$). Phép biến đổi phi tuyến này giúp mạng nơ-ron nhận diện tốt hơn những sự cải thiện hoặc sa sút học lực cực đoan.
- **Chỉ báo Quỹ đạo Điểm số (Grade Trajectory):** Tính toán "Động lượng học tập" (Momentum) thông qua gia tốc điểm số ($G2 - G1$) và tỷ lệ tăng trưởng ($G2 / (G1 + 1)$). Đặc trưng động học này cung cấp cho BiLSTM một tín hiệu chuỗi rõ ràng hơn về độ ổn định của học sinh.

## 4.3. Đề xuất Kiến trúc Mạng Multi-Scale CNN-BiLSTM
Khác với thiết kế chuỗi thẳng thông thường, kiến trúc đề xuất của chúng tôi áp dụng cơ chế xử lý đa tỷ lệ (Multi-Scale). Đầu vào không gian bảng được cung cấp đồng thời cho ba nhánh CNN 1D với các kích thước bộ lọc (Kernel sizes) khác nhau:
- **Nhánh Kernel nhỏ (size 3):** Quét các tương quan đặc trưng gần gũi.
- **Nhánh Kernel trung bình (size 5):** Tìm kiếm các nhóm đặc trưng hành vi phức tạp hơn.
- **Nhánh BiLSTM độc lập:** Một bản sao nguyên thủy của dữ liệu được truyền thẳng vào một nhánh LSTM riêng để chống mất mát thông tin.

Tín hiệu xuất ra từ ba nhánh CNN được tổng hợp bằng kỹ thuật Global Average Pooling, sau đó ghép (Concatenate) với tín hiệu dọc và đi vào 2 tầng khối mạng BiLSTM. Tầng BiLSTM (với kích thước ẩn lên tới 192/256 nút) làm nhiệm vụ nhào nặn khối tín hiệu này, ánh xạ chúng vào không gian không gian quyết định. Cuối cùng, tầng Fully Connected với cấu trúc Dropout (20-30%) đưa ra phân phối xác suất dự đoán thông qua hàm Softmax.

## 4.4. Giao thức Đánh giá Tối Mật (Strict Validation Protocol 70/15/15)
Để quét sạch hoàn toàn hiện tượng Rò rỉ dữ liệu (Data Leakage) đã bóc trần ở Chương 3, chúng tôi xác lập một quy tắc thực nghiệm nghiêm ngặt bất khả xâm phạm:
1. **Chia tách 3 phần:** Tập dữ liệu gốc được trộn ngẫu nhiên và phân tách thành Tập Huấn luyện (Train) 70%, Tập Xác thực (Validation) 15%, và Tập Kiểm thử (Test) 15%. Phân tách đảm bảo tính phân tầng (Stratified) để bảo toàn tỷ lệ nhãn rớt môn.
2. **Cách ly Tập Test:** Trong suốt quá trình mạng học sâu huấn luyện, vòng lặp tối ưu hóa siêu tham số (Optuna), cũng như chiến lược Dừng sớm (Early Stopping), mô hình **TUYỆT ĐỐI KHÔNG** được phép đánh giá trên tập Test. Tập Xác thực (Validation) sẽ đóng vai trò như một phòng thi thử.
3. **Chốt mô hình (Model Freeze):** Khi và chỉ khi kiến trúc mạng, siêu tham số và epoch dừng đã được chốt hoàn toàn thông qua thành tích trên tập Validation, mô hình mới được mở khóa tập Test. Đánh giá trên tập Test diễn ra **một lần duy nhất**. Kết quả trả về là kết luận khoa học cuối cùng, dập tắt mọi ảo tưởng do quá khớp sinh ra.
""")
print('Appended')
