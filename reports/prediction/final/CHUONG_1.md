# Chương 1. Tổng quan đề tài nghiên cứu

## 1.1. Lý do chọn đề tài

### 1.1.1. Nhu cầu cảnh báo sớm nguy cơ học tập

Nhà trường cần nhận diện sinh viên có nguy cơ không hoàn thành môn học khi vẫn còn thời gian hỗ trợ, chứ không phải sau khi đã có điểm cuối kỳ hoặc trạng thái hủy đăng ký. Đây là bài toán cảnh báo sớm trong khai phá dữ liệu giáo dục: mọi đặc trưng chỉ được lấy từ thông tin đã quan sát trước một mốc thời gian cho trước.

Hai nguồn dữ liệu công khai thường được dùng trong lĩnh vực này khác nhau về bản chất thời gian. Bộ UCI Student Performance (Cortez và Silva, 2008) gồm 395 bản ghi môn Toán và 649 bản ghi môn Tiếng Bồ Đào Nha, gộp thành 1.044 cặp học sinh–môn. Chuỗi điểm tối đa hai bước (G1, G2); nhãn được tạo từ điểm cuối kỳ G3 theo ngưỡng 10 trên thang 0–20. Tỷ lệ lớp nguy cơ là 0,220 (230/1.044). Bộ OULAD (Kuzilek, Hlosta và Zdrahal, 2017) gồm 32.593 lượt ghi danh, 28.785 sinh viên và hơn 10 triệu sự kiện tương tác trên môi trường học trực tuyến. Nhãn nguy cơ là Fail hoặc Withdrawn. Năm mốc quan sát được đặt tại 20%, 35%, 50%, 75% và 100% chiều dài môn học.

Nếu đưa điểm cuối kỳ, kết quả cuối môn, điểm bài kiểm tra hoặc ngày hủy đăng ký vào đầu vào thì mô hình không còn mang ý nghĩa cảnh báo sớm.

### 1.1.2. Hạn chế của độ chính xác tổng thể khi lớp nguy cơ là thiểu số

Lớp nguy cơ thường chiếm tỷ lệ thấp. Trên UCI, một mô hình luôn dự đoán không nguy cơ đã đạt khoảng 0,78 độ chính xác tổng thể mà không xếp hạng được ai cần hỗ trợ. Hệ quả là bỏ sót sinh viên đang có nguy cơ (độ nhạy thấp) hoặc cảnh báo tràn lan (độ chính xác dương thấp). Một ngưỡng quyết định cố định cho mọi mốc thông tin cũng làm sai lệch ý nghĩa vận hành, vì cùng một xác suất được diễn giải khác nhau khi chưa có điểm giữa kỳ hoặc khi chuỗi tương tác còn rất ngắn.

Do đó đề tài lấy Average Precision (AP) làm chỉ số chính. AP đo chất lượng xếp hạng lớp dương trên toàn bộ ngưỡng, phù hợp hơn ROC-AUC khi hai lớp không cân bằng.

### 1.1.3. Nhu cầu một kiến trúc dùng chung trên hai miền dữ liệu

Hai miền không được gộp thành một tập huấn luyện. UCI là bản ghi theo học kỳ, chuỗi dài tối đa hai bước; OULAD là chuỗi theo tuần, có thể dài tới gần 40 bước. Mô hình chỉ dựa trên bảng đặc trưng phẳng bỏ qua thứ tự điểm và thứ tự tuần tương tác. Ngược lại, mô hình chỉ dựa trên CNN hay LSTM thiếu tín hiệu khi chuỗi còn rỗng, chẳng hạn khi chưa có G1 và G2.

Đề tài cần một kiến trúc nhận cùng kiểu đầu vào trên cả hai miền, tắt nhánh chuỗi khi chưa có quan sát thời gian, và dùng một mô hình đã huấn luyện cho mỗi miền để chấm mọi mốc thông tin. Trên OULAD, sự kiện chỉ được lấy khi thời điểm xảy ra nằm trước mốc cắt.

### 1.1.4. Tiềm năng của kiến trúc lai CNN–BiLSTM và module khuyến nghị

CNN một chiều trích mẫu cục bộ trên cửa sổ ngắn. BiLSTM mã hóa phụ thuộc hai chiều trong cửa sổ đã cắt tại mốc quan sát. Nhánh bảng giữ ngữ cảnh tĩnh và thống kê gộp. Cổng softmax ba nhánh điều khiển tỷ lệ đóng góp của từng thành phần, trong đó nhánh chuỗi bị tắt khi chưa có bước thời gian hợp lệ.

Xác suất nguy cơ chưa đủ để hỗ trợ học tập. Nhà trường còn cần các hành động khả thi như hoàn thành bài đánh giá, phục hồi tương tác hay ôn luyện đều đặn. Module khuyến nghị xếp hạng các hành động đó trên cơ sở xác suất nguy cơ và bằng chứng đã quan sát, không ước lượng hiệu ứng nhân quả lên kết quả cuối môn.

Phạm vi khóa luận là mô hình dự đoán, đánh giá thực nghiệm và module khuyến nghị. Đề tài không xây dựng giao diện người dùng.

## 1.2. Mục tiêu nghiên cứu

### 1.2.1. Mục tiêu tổng quát

Xây dựng và đánh giá mô hình Hybrid CNN–BiLSTM dự đoán nguy cơ học tập nhị phân trên UCI và OULAD, đồng thời gắn module khuyến nghị hành động hỗ trợ trên các mốc 20%, 35%, 50% và 75% của OULAD.

### 1.2.2. Mục tiêu cụ thể

- Về mô hình dự báo:
  - Một kiến trúc Hybrid CNN–BiLSTM dùng chung cho hai miền, khác nhau chủ yếu ở chiều đầu vào và trọng số đã học.
  - Một mô hình UCI chấm các mốc S0, S1, S2; một mô hình OULAD chấm các mốc 20%, 35%, 50%, 75% và 100%.
  - CNN chạy song song với BiLSTM, kết hợp bằng cổng softmax ba nhánh; tắt CNN và BiLSTM khi chưa có chuỗi.
- Về đánh giá:
  - Chỉ số chính là AP trên tập kiểm định trong, trung bình 9 lần chạy (3 phần chia nhóm × 3 hạt giống).
  - So sánh với hồi quy logistic, cây quyết định, rừng ngẫu nhiên, SVM, mạng perceptron đa lớp và XGBoost trên cùng quy trình đánh giá.
  - Phần chia ngoài không dùng để chọn kiến trúc hay siêu tham số.
  - Kiểm định: (H1) trên UCI S1 và OULAD 35%, AP của mô hình đầy đủ cao hơn nhánh bảng đơn thuần; (H2) trên các mốc đã có chuỗi, AP của Hybrid cao hơn hồi quy logistic và rừng ngẫu nhiên; (H3) khối lượng cổng dịch sang CNN và BiLSTM khi mốc OULAD tăng.
- Về module khuyến nghị:
  - Chỉ áp dụng trên OULAD 20–75%; mốc 100% không đưa vào khuyến nghị.
  - Đánh giá bằng NDCG@3 trên tập kiểm định độc lập; tỷ lệ hành động không hợp lệ bằng 0.
  - Không ước lượng hiệu ứng can thiệp lên kết quả cuối môn.

Mốc công bố chính là UCI S1, S2 và OULAD từ 35% đến 75%. S0 và 20% là mốc thiếu chuỗi, không dùng để bác bỏ kiến trúc lai. Số liệu trình bày ở Chương 4.

## 1.3. Đối tượng và phạm vi nghiên cứu

### 1.3.1. Đối tượng nghiên cứu

- Dữ liệu đầu vào:
  - UCI: đặc trưng nền, chuỗi G1/G2 theo mốc S0/S1/S2, thống kê gộp trên điểm đã quan sát. Điểm G3 và số buổi vắng không dùng làm đầu vào.
  - OULAD: đặc trưng nền, 11 kênh chuỗi theo tuần, 13 thống kê gộp tại mốc cắt. Kết quả cuối môn, điểm bài kiểm tra và ngày hủy đăng ký không dùng làm đầu vào. Lượt ghi danh đã hủy trước mốc cắt bị loại khỏi mốc đó.
- Biến mục tiêu:
  - UCI: nguy cơ khi G3 nhỏ hơn 10.
  - OULAD: nguy cơ khi kết quả là Fail hoặc Withdrawn.
  - Đầu ra mô hình: xác suất nguy cơ, ngưỡng quyết định chọn trên tập dừng, nhãn vận hành và độ bất định nhị phân.
- Mô hình:
  - Mô hình dự đoán: Hybrid CNN–BiLSTM.
  - Các mô hình đối sánh: hồi quy logistic, cây quyết định, rừng ngẫu nhiên, SVM, mạng perceptron đa lớp, XGBoost.
  - Module khuyến nghị: năm mô hình boosting diễn giải được, kèm luật khả thi cứng.

### 1.3.2. Phạm vi nghiên cứu

- Không gian và thời gian dữ liệu:
  - UCI: 1.044 bản ghi, 662 nhóm học sinh.
  - OULAD: số bản ghi còn đủ điều kiện theo mốc khoảng 26.697 (20%) đến 22.522 (100%).
  - Không sử dụng dữ liệu sinh viên tại Việt Nam.
- Phạm vi kỹ thuật:
  - Học sâu bằng PyTorch; chuẩn hóa và trọng số lớp dương chỉ ước lượng trên tập huấn luyện.
  - Đánh giá theo nhóm, không xáo trộn độc lập từng dòng.
  - Module khuyến nghị trên OULAD 20–75%, tập kiểm định độc lập 632 tình huống.
- Ngoài phạm vi:
  - Giao diện người dùng và dịch vụ API công khai.
  - Dùng phần chia ngoài để chọn mô hình.
  - Ước lượng hiệu ứng can thiệp; thử nghiệm với giảng viên.

## 1.4. Phương pháp nghiên cứu

### 1.4.1. Phương pháp nghiên cứu lý thuyết

Đề tài tổng hợp tài liệu về khai phá dữ liệu giáo dục, cảnh báo sớm, rò rỉ thông tin, lệch lớp, CNN một chiều, BiLSTM, cơ chế cổng kết hợp và xếp hạng hành động. Trên cơ sở đó, kiến trúc lai và module khuyến nghị được thiết kế sao cho đầu vào luôn nằm trước mốc quan sát.

### 1.4.2. Phương pháp nghiên cứu thực nghiệm

Dữ liệu được tái cấu trúc thành tensor có mặt nạ, chuẩn hóa trên tập huấn luyện, chia nhóm học sinh thành tập huấn luyện, tập dừng và tập kiểm định trong. Mô hình được huấn luyện bằng entropi chéo nhị phân có trọng số, dừng sớm theo AP trên tập dừng, chọn ngưỡng trên tập dừng rồi báo cáo trên tập kiểm định. Mỗi mốc có 9 lần chạy. Module khuyến nghị được đánh giá trên tập độc lập, không hiệu chỉnh trên tập đó.

## 1.5. Ý nghĩa khoa học và thực tiễn của đề tài

### 1.5.1. Ý nghĩa khoa học

Đề tài đưa ra một cách tổ chức đầu vào thống nhất cho hai miền khác độ dài chuỗi, kèm cơ chế tắt nhánh khi chưa có quan sát thời gian. Việc dùng AP và kiểm định trên nhiều lần chạy giúp đánh giá xếp hạng lớp thiểu số một cách thận trọng hơn độ chính xác tổng thể.

### 1.5.2. Ý nghĩa thực tiễn

Mô hình cung cấp xác suất nguy cơ tại các mốc còn thời gian hỗ trợ (S1 trên UCI; 35–75% trên OULAD). Module khuyến nghị chuyển xác suất thành hành động có điều kiện khả thi. Kết quả không được diễn giải như đã triển khai tại một cơ sở đào tạo cụ thể ở Việt Nam.

## 1.6. Bố cục của khóa luận

Chương I trình bày lý do, mục tiêu, đối tượng, phạm vi, phương pháp và ý nghĩa.

Chương II trình bày cơ sở lý thuyết về nguy cơ học tập, dữ liệu, tiền xử lý, kiến trúc học sâu, khuyến nghị và chỉ số đánh giá.

Chương III trình bày phân tích dữ liệu, thiết kế tiền xử lý, kiến trúc, quy trình huấn luyện và module khuyến nghị, không đưa kết quả thực nghiệm.

Chương IV trình bày môi trường, quá trình huấn luyện, kết quả dự đoán, đối sánh, trực quan hóa và kết quả module khuyến nghị.

Chương V trình bày kết luận, hạn chế và hướng phát triển.
