# Chương 2. Cơ sở lý thuyết

## 2.1. Cơ sở lý thuyết về cảnh báo sớm nguy cơ học tập

### 2.1.1. Định nghĩa bài toán

Cảnh báo sớm trong khai phá dữ liệu giáo dục là bài toán phân lớp nhị phân có ràng buộc thời gian. Tại một mốc quan sát, mô hình ước lượng xác suất sinh viên thuộc lớp nguy cơ, chỉ dùng thông tin đã có trước mốc đó.

Lớp dương trên UCI là G3 nhỏ hơn 10 (thang 0–20). Lớp dương trên OULAD là Fail hoặc Withdrawn. Đầu ra là một xác suất, không phải hồi quy điểm số liên tục.

### 2.1.2. Lệch lớp và lựa chọn chỉ số

Khi lớp dương là thiểu số, độ chính xác tổng thể dễ bị thống trị bởi lớp âm. ROC-AUC đối xử hai lớp tương đối đối xứng, có thể cao trong khi độ chính xác dương vẫn thấp. Average Precision đo diện tích dưới đường precision–recall theo thứ tự xác suất giảm dần, nên phù hợp hơn để đánh giá khả năng xếp hạng sinh viên nguy cơ.

Precision, Recall và F1 chỉ được báo cáo tại một ngưỡng đã chọn trên tập dừng, không thay thế AP.

### 2.1.3. Rò rỉ nhãn và rò rỉ thời gian

Rò rỉ nhãn xảy ra khi biến dùng để tạo nhãn, hoặc biến đồng thời với nhãn, được đưa vào đầu vào. Điểm G3, kết quả cuối môn và số buổi vắng thuộc nhóm này. Rò rỉ thời gian xảy ra khi sự kiện sau mốc cắt vẫn được tính, hoặc khi lượt ghi danh đã hủy trước mốc cắt vẫn được giữ như một mẫu cảnh báo sớm.

Trên OULAD, sự kiện chỉ được lấy khi thời điểm nằm trong khoảng từ lúc bắt đầu quan sát đến trước mốc cắt. Lượt ghi danh hủy trước mốc cắt bị loại khỏi mốc đó.

### 2.1.4. Mốc thông tin

UCI có ba cách nhìn cùng một bản ghi: S0 chưa có điểm giữa kỳ, S1 đã có G1, S2 đã có G1 rồi G2. OULAD có năm cách nhìn theo 20%, 35%, 50%, 75% và 100% chiều dài môn. Đây là các trạng thái thông tin của cùng một mô hình, không phải năm mô hình độc lập. Khi chuỗi rỗng, nhánh thời gian phải được tắt.

## 2.2. Hai bộ dữ liệu sử dụng trong đề tài

### 2.2.1. UCI Student Performance

Cortez và Silva (2008) công bố hai tập điểm theo học kỳ, mỗi dòng là một cặp học sinh–môn. Điểm G1, G2, G3 nằm trên thang 0–20. Đề tài gộp hai môn thành 1.044 bản ghi. Việc chia tập theo dòng có thể đưa cùng một học sinh vào cả tập huấn luyện lẫn tập kiểm định, vì 366 nhóm xuất hiện ở cả hai môn. Do đó việc chia được thực hiện theo nhóm học sinh.

Chuỗi tối đa hai bước, thích hợp để kiểm tra hành vi mô hình khi chưa có điểm.

### 2.2.2. OULAD

Kuzilek, Hlosta và Zdrahal (2017) công bố dữ liệu Đại học Mở, gồm thông tin ghi danh và nhật ký tương tác theo ngày. Kết quả cuối môn gồm Pass, Distinction, Fail và Withdrawn. Chiều dài môn khác nhau theo khóa học, nên mốc cắt là tỷ lệ chiều dài chứ không phải số tuần cố định.

Tại mốc 100%, nhiều lượt rút sớm đã bị loại, tỷ lệ lớp dương giảm so với mốc 20%. AP tại 100% không được đọc như chỉ số cảnh báo sớm.

Hai miền không gộp huấn luyện. AP trên UCI và AP trên OULAD không so trực tiếp vì khác tỷ lệ lớp và khác cách sinh dữ liệu.

## 2.3. Khai phá dữ liệu giáo dục

Khai phá dữ liệu tìm mẫu có ích từ dữ liệu lớn. Các nhiệm vụ điển hình gồm phân lớp, hồi quy, phân cụm và luật kết hợp. Đề tài thuộc phân lớp nhị phân có ràng buộc thời gian, không phải hồi quy điểm cuối kỳ.

## 2.4. Dữ liệu tuần tự có mặt nạ

Khác với dự báo chuỗi hồi quy (giá trị bước kế tiếp của chính chuỗi), mục tiêu ở đây là nhãn cuối kỳ hoặc cuối môn. Độ dài chuỗi thay đổi theo mốc và theo sinh viên, nên cần mặt nạ thời gian. Hướng ngược của BiLSTM chỉ được phép chạy trong cửa sổ đã quan sát, không đọc sự kiện sau mốc cắt.

Vì vậy CNN và BiLSTM được bố trí song song trên cùng chuỗi đã che, rồi kết hợp với nhánh bảng, khác với sơ đồ CNN nối tiếp LSTM thường gặp trong dự báo chuỗi hồi quy.

## 2.5. Tiền xử lý dữ liệu

Các bước chính gồm: loại biến rò rỉ; dựng chuỗi có mặt nạ; chuẩn hóa trung bình và phương sai chỉ trên tập huấn luyện; mã hóa biến phân loại trên tập huấn luyện; tính trọng số lớp dương trên tập huấn luyện. Ô không hợp lệ được gán 0 và mặt nạ 0, không nội suy điểm G1, G2 giả và không nội suy tuần tương tác sau mốc cắt.

Chia tập theo nhóm học sinh. Phần chia ngoài chỉ dùng để loại định danh, không tham gia chọn mô hình.

## 2.6. Các mô hình học sâu sử dụng trong đề tài

CNN một chiều dùng bộ lọc trượt theo thời gian để nhận diện mẫu cục bộ. Trong mô hình đề xuất, chuỗi được chiếu về 128 chiều, qua hai khối dư với 64 kênh, kích thước hạt 2, giãn nở 1 rồi 2, sau đó gộp theo trung bình và cực đại có mặt nạ.

LSTM dùng các cổng để giảm suy giảm gradient so với mạng hồi quy cổ điển. BiLSTM chạy hai hướng trên chuỗi đã cắt. Đề tài dùng một lớp BiLSTM, kích thước ẩn 128, đóng gói theo độ dài thực để không học phần đệm.

Nhánh bảng chiếu đặc trưng tĩnh và thống kê gộp về cùng không gian 128 chiều. Cổng softmax nhận biểu diễn ba nhánh cùng cờ hiện diện và mức tiến độ môn, gán xác suất khối lượng; nhánh không hiện diện nhận khối lượng 0. Đầu ra là một logit nhị phân, xác suất thu được bằng hàm sigmoid.

Các mô hình học máy truyền thống (hồi quy logistic, cây quyết định, rừng ngẫu nhiên, SVM, mạng perceptron đa lớp, XGBoost) được huấn luyện trên cùng các đặc trưng bảng đã tóm tắt để đối sánh, không nhìn thứ tự thô của chuỗi.

## 2.7. Cơ sở lý thuyết của module khuyến nghị

Bài toán khuyến nghị ở đây là xếp hạng hành động hỗ trợ khả thi, không phải gợi ý môn học hay tài liệu theo lọc cộng tác. Đầu vào là xác suất nguy cơ, độ bất định và các bằng chứng đã quan sát (mức tương tác, hạn bài, phạm vi nội dung). Explainable Boosting Machine cho phép mô hình hóa quan hệ cộng tính, thuận tiện hơn mạng sâu khi cần diễn giải mức đóng góp của từng bằng chứng.

Tính khả thi được kiểm bằng luật cứng: một hành động không được xếp nếu thiếu điều kiện tối thiểu, ví dụ không còn bài chưa nộp thì không khuyến nghị hoàn thành bài đánh giá. Cơ chế từ chối xuất hiện khi xác suất dưới ngưỡng, khi độ bất định cao, hoặc khi không còn hành động hợp lệ.

## 2.8. Huấn luyện và đánh giá

Hàm mất mát là entropi chéo nhị phân trên logit, có trọng số lớp dương. Bộ tối ưu là AdamW. Ngưỡng quyết định được chọn trên tập dừng theo F1, sau đó đến độ nhạy, rồi độ gần 0,5. AP được tính theo định nghĩa average precision của scikit-learn.

Đánh giá không dùng kiểm định chéo xáo trộn độc lập từng dòng. Kết quả mỗi mốc là trung bình 9 lần chạy. Module khuyến nghị dùng NDCG tại 3, Precision tại 1 và tỷ lệ hành động không hợp lệ.
