# Chương 5. Kết luận và hướng phát triển

## 5.1. Kết luận

Khóa luận tốt nghiệp đã hoàn thành các mục tiêu đề ra, tập trung vào việc nghiên cứu và xây dựng một mô hình học sâu lai để dự đoán nguy cơ học tập nhị phân trên hai bộ dữ liệu UCI và OULAD, sau đó gắn module khuyến nghị hành động hỗ trợ trên các mốc còn thời gian can thiệp của OULAD.

Qua quá trình nghiên cứu và thực nghiệm, đề tài đã đạt được những kết quả cụ thể sau:

- Xây dựng thành công mô hình học sâu lai: Đã xây dựng và huấn luyện Hybrid CNN–BiLSTM bằng PyTorch, kết hợp nhánh bảng, CNN một chiều và BiLSTM qua cổng softmax ba nhánh. Kiến trúc này được lựa chọn nhằm tận dụng khả năng mã hóa ngữ cảnh của nhánh bảng, khả năng trích mẫu cục bộ của CNN và khả năng nắm bắt phụ thuộc thời gian của BiLSTM. Một mô hình dùng cho UCI (S0, S1, S2) và một mô hình dùng cho OULAD (20% đến 100%). Khi chưa có chuỗi, CNN và BiLSTM được tắt.

- Đánh giá hiệu năng mô hình một cách khách quan: Đã tiến hành thực nghiệm theo chia nhóm, chín lần chạy cho mỗi mốc, chỉ số chính là Average Precision. Trên UCI, AP đạt 0,821 tại S1 và 0,910 tại S2. Trên OULAD, AP tăng từ 0,762 tại 20% lên 0,920 tại 100% trên cùng một mô hình. Từ mốc 35% trở đi, Hybrid CNN–BiLSTM cao hơn hồi quy logistic và rừng ngẫu nhiên theo kiểm định Wilcoxon. Thí nghiệm loại bỏ thành phần cho thấy mô hình đầy đủ không kém các biến thể chỉ dùng một nhánh. Khối lượng cổng dịch sang BiLSTM khi chuỗi dài hơn, phù hợp giả thuyết về cơ chế kết hợp.

- Xây dựng module khuyến nghị: Module xếp hạng năm hành động hỗ trợ trên xác suất nguy cơ và bằng chứng đã quan sát. Trên tập độc lập 632 tình huống, NDCG tại 3 đạt 0,888, không phát hành hành động vi phạm luật khả thi, đồng thời phủ đủ năm hành động ở vị trí đứng đầu.

Từ các kết quả trên, khóa luận mang lại những đóng góp về học thuật và thực tiễn:

- Về học thuật: Đề tài áp dụng kiến trúc lai CNN–BiLSTM có cổng điều kiện cho bài toán cảnh báo sớm trên hai miền khác độ dài chuỗi, với quy trình đánh giá theo nhóm và chỉ số xếp hạng lớp thiểu số. Toàn bộ các bước từ loại biến rò rỉ, chuẩn hóa trên tập huấn luyện đến chín lần chạy được trình bày tường minh, có thể tham khảo cho các nghiên cứu tương tự.

- Về thực tiễn: Mô hình cung cấp xác suất nguy cơ tại các mốc còn thời gian hỗ trợ. Module khuyến nghị chuyển xác suất thành hành động có điều kiện khả thi, thay vì chỉ dừng ở một con số rủi ro. Kết quả không được hiểu như đã triển khai vận hành tại một cơ sở đào tạo cụ thể.

## 5.2. Hạn chế

Mặc dù khóa luận đã đạt được những mục tiêu chính đề ra, vẫn còn tồn tại một số hạn chế nhất định cần được nhìn nhận một cách khách quan. Việc xác định rõ các hạn chế này sẽ là cơ sở quan trọng cho các hướng phát triển trong tương lai.

- Hạn chế về dữ liệu:
  - Quy mô và độ dài chuỗi: Bộ UCI chỉ gồm 1.044 bản ghi, chuỗi tối đa hai bước. Đối với học sâu, đây là kích thước tương đối nhỏ. Tại S0 chưa có điểm giữa kỳ, khoảng cách giữa tập huấn luyện và tập kiểm định còn lớn (0,125) và sai số hiệu chỉnh còn cao (0,254), hạn chế khả năng học mẫu phức tạp.
  - Phạm vi đặc trưng: Mô hình chưa sử dụng dữ liệu sinh viên tại Việt Nam. Các yếu tố như phản hồi giảng viên, hoàn cảnh kinh tế ngoài các biến có sẵn, hay tương tác diễn đàn ở mức ngữ nghĩa chưa được đưa vào.
  - Mốc 100% của OULAD: Nhiều lượt rút sớm bị loại khỏi mẫu, tỷ lệ lớp dương giảm so với mốc 20%. AP tại 100% không phản ánh khả năng cảnh báo sớm.

- Hạn chế về mô hình và kỹ thuật:
  - Kiến trúc: Mặc dù Hybrid CNN–BiLSTM đã cho thấy hiệu quả trên các mốc có chuỗi, đây vẫn là một kiến trúc tương đối gọn. Các kiến trúc như Transformer với cơ chế Attention, vốn đang thành công trên chuỗi dài, chưa được thử trên cùng cách chia tập và cùng chỉ số AP.
  - Tối ưu hóa siêu tham số: Tốc độ học, Dropout và kích thước lô được chọn theo miền trên cơ sở thử nghiệm có kiểm soát, chưa áp dụng tối ưu siêu tham số tự động một cách hệ thống trên toàn bộ không gian tìm kiếm.
  - Khả năng diễn giải: Việc đọc cổng trung bình theo mốc giúp hiểu nhánh nào được dùng, nhưng chưa giải thích được từng dự đoán cụ thể theo từng đặc trưng gốc như các kỹ thuật SHAP hay LIME.
  - Phần chia ngoài: Phần chia ngoài không được mở khi công bố kết quả, nên chưa có ước lượng trên tập hoàn toàn tách khỏi quá trình chọn mô hình.

- Hạn chế về phạm vi ứng dụng:
  - Tính đặc thù dữ liệu: Mô hình được huấn luyện trên UCI và OULAD. Do đặc điểm tổ chức đào tạo và hành vi học tập khác nhau, mô hình có thể không giữ nguyên hiệu suất nếu áp trực tiếp cho một trường khác mà không huấn luyện lại.
  - Module khuyến nghị: Module xếp hạng hành động khả thi, không ước lượng hiệu ứng can thiệp lên kết quả cuối môn. Chưa có thử nghiệm với cố vấn học tập. Fail và Withdrawn đang được gộp chung một lớp dương, trong khi hai nhóm này có thể cần hành động khác nhau.

## 5.3. Hướng phát triển trong tương lai

Từ những hạn chế đã được phân tích, đề tài mở ra một số hướng nghiên cứu nhằm xây dựng hệ thống cảnh báo và hỗ trợ học tập đầy đủ hơn.

- Nâng cao chất lượng và quy mô dữ liệu:
  - Thu thập dữ liệu của cơ sở đào tạo tại Việt Nam, giữ nguyên nguyên tắc không dùng nhãn và không dùng sự kiện sau mốc cắt.
  - Mở phần chia ngoài một lần sau khi đã đóng băng mọi lựa chọn, để có ước lượng khách quan hơn.
  - Tách Fail và Withdrawn thành hai bài toán nếu số mẫu tại tập kiểm định đủ lớn.

- Cải tiến mô hình:
  - Thử Transformer hoặc mạng nơ-ron đồ thị theo thời gian trên OULAD từ 35% đến 75%, cùng cách chia tập và cùng AP với mô hình hiện tại.
  - Xem xét bỏ thống kê gộp trùng với điểm giữa kỳ trên UCI nếu biến thể chỉ dùng chuỗi không kém mô hình hiện tại trên chín lần chạy.
  - Hiệu chỉnh xác suất, đặc biệt tại S0, nơi sai số hiệu chỉnh còn 0,254.
  - Bổ sung kỹ thuật diễn giải cục bộ để chỉ ra đặc trưng nào đẩy một sinh viên vào nhóm nguy cơ.

- Hoàn thiện khuyến nghị và triển khai:
  - Cá nhân hóa hành động theo môn học và theo tiến độ, không hiệu chỉnh module trên tập đánh giá độc lập.
  - Thử nghiệm với cố vấn học tập, đo mức hữu ích của gợi ý, không suy ra hiệu quả nhân quả lên điểm cuối.
  - Nếu triển khai sản phẩm, có thể xây dựng giao diện cảnh báo sớm; phần này nằm ngoài phạm vi khóa luận hiện tại.
