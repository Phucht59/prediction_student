# BÁO CÁO KHÓA LUẬN TỐT NGHIỆP
## XÂY DỰNG MÔ HÌNH HỌC SÂU KẾT HỢP ĐỂ DỰ ĐOÁN THÀNH TÍCH HỌC TẬP SINH VIÊN

---

# CHƯƠNG 1: MỞ ĐẦU

## 1.1. Đặt vấn đề
Sự phát triển mạnh mẽ của công nghệ thông tin và truyền thông trong thập kỷ qua đã tạo ra một cuộc cách mạng trong lĩnh vực giáo dục. Các hệ thống quản lý học tập (Learning Management Systems - LMS), các nền tảng học trực tuyến (E-learning) và các hồ sơ học bạ điện tử đã lưu trữ một khối lượng dữ liệu khổng lồ. Khối lượng dữ liệu giáo dục (Educational Data) này không chỉ bao gồm thông tin cá nhân cơ bản, mà còn chứa đựng các chuỗi hành vi học tập chi tiết như tần suất đăng nhập, số lần xem tài liệu, điểm số các bài kiểm tra thành phần, và thời gian tương tác với hệ thống. Việc khai thác khối tài nguyên này đã khai sinh ra lĩnh vực Khai phá dữ liệu giáo dục (Educational Data Mining - EDM).

Trong EDM, dự đoán thành tích học tập của sinh viên là một trong những bài toán mang tính sống còn. Nếu có thể dự đoán sớm khả năng học tập của sinh viên, các cơ sở giáo dục có thể kịp thời đưa ra các cảnh báo, cung cấp sự hỗ trợ học thuật, và điều chỉnh phương pháp giảng dạy để ngăn chặn nguy cơ trượt môn hoặc bỏ học. Truyền thống, bài toán này được giải quyết bằng các mô hình học máy thống kê như Hồi quy Logistic, Cây quyết định (Decision Tree) hay Rừng ngẫu nhiên (Random Forest). Tuy nhiên, khi đối mặt với dữ liệu có tính chiều cao và cấu trúc phức tạp, các mô hình học máy truyền thống thường gặp khó khăn trong việc trích xuất các đặc trưng tiềm ẩn.

Gần đây, Học sâu (Deep Learning) đã được ứng dụng mạnh mẽ vào bài toán EDM và cho thấy những kết quả ấn tượng. Đặc biệt, kiến trúc lai kết hợp giữa Mạng Nơ-ron Tích chập (CNN) và Mạng Bộ nhớ Ngắn hạn Hai chiều (BiLSTM) đã được một số nghiên cứu chứng minh là mang lại hiệu năng dự đoán xuất sắc. Mặc dù vậy, khi tiến hành thẩm định và mô phỏng lại các nghiên cứu này, chúng tôi phát hiện ra một vấn đề nghiêm trọng đe dọa đến tính ứng dụng thực tiễn của các mô hình học sâu trong giáo dục: **Hiện tượng Rò rỉ dữ liệu (Data Leakage)** và **Sự ảo giác của các chỉ số hiệu năng (Overfitting Evaluation)**. 

Phần lớn các nghiên cứu hiện tại báo cáo chỉ số F1-score lên tới 0.90 – 1.00 trên những bộ dữ liệu cực nhỏ (dưới 1000 sinh viên) với sự phân bố nhãn cực kỳ mất cân bằng. Sự thật đằng sau những con số này là do tập kiểm thử (Test set) đã bị rò rỉ vào quá trình huấn luyện thông qua kỹ thuật Dừng sớm (Early Stopping) – mô hình liên tục "nhìn trộm" tập kiểm thử để chọn ra thời điểm nó đạt kết quả cao nhất. Điều này khiến mô hình mất đi khả năng tổng quát hóa (Generalization) khi triển khai thực tế.

Nhận thức được khoảng trống nghiên cứu này, khóa luận tiến hành xây dựng một mô hình học sâu lai CNN-BiLSTM, kết hợp với các siêu đặc trưng (Ultra Engineered Features), và đánh giá nó dưới một **Giao thức Kiểm chứng Nghiêm ngặt (Strict Validation Protocol)** hoàn toàn độc lập. Nghiên cứu nhằm mục đích mang lại một cái nhìn trung thực, khoa học và thực tiễn về năng lực thực sự của AI trong dự đoán điểm số sinh viên.

## 1.2. Mục tiêu nghiên cứu
Mục tiêu tổng quát của khóa luận là xây dựng, tối ưu hóa và đánh giá một kiến trúc học sâu lai (Hybrid Deep Learning) đảm bảo tính chính xác và độ tin cậy tuyệt đối trong việc dự đoán kết quả học tập của học viên. Cụ thể, các mục tiêu chi tiết bao gồm:
1. **Xây dựng kiến trúc lai:** Thiết kế mô hình Multi-Scale CNN-BiLSTM nhằm khai thác tối đa đặc trưng cục bộ (thông qua CNN) và quan hệ theo chuỗi (thông qua BiLSTM) từ dữ liệu bảng (Tabular data) của sinh viên.
2. **Khai phá đặc trưng chuyên sâu (Feature Engineering):** Đề xuất và triển khai các kỹ thuật trích xuất đặc trưng đa thức (Polynomial Features) và phân tích quỹ đạo điểm số (Grade Trajectory) nhằm bù đắp cho kích thước dữ liệu hạn chế.
3. **Phát triển giao thức Strict Validation:** Xây dựng quy trình đánh giá 3 tập độc lập (Train/Validation/Test với tỷ lệ 70/15/15), loại bỏ hoàn toàn hiện tượng Data Leakage.
4. **Đánh giá Phương sai dữ liệu (Data Variance):** Thực hiện phân tích đánh giá chéo đa hạt giống (Multi-seed Evaluation) để đo lường giới hạn đỉnh cao (Peak F1) và sức mạnh thực chất (Average F1) của mô hình trên các tập dữ liệu nhỏ.

## 1.3. Đối tượng và phạm vi nghiên cứu
- **Đối tượng nghiên cứu:** Hành vi học tập, kết quả học tập, và các mô hình học sâu (CNN, BiLSTM).
- **Phạm vi dữ liệu:** Nghiên cứu sử dụng 3 tập dữ liệu giáo dục công khai bao gồm Student Performance (môn Toán và tiếng Bồ Đào Nha) thu thập từ các trường trung học ở Bồ Đào Nha, và tập dữ liệu xAPI Educational Data thu thập từ một hệ thống quản lý học tập.
- **Phạm vi kỹ thuật:** Ứng dụng ngôn ngữ lập trình Python, thư viện học sâu PyTorch, và thư viện tối ưu hóa Optuna để tiến hành huấn luyện và đánh giá mô hình. Bài toán được mô hình hóa dưới dạng bài toán phân lớp đa nhãn (Multi-class Classification) tương ứng với các phân cấp điểm số.

## 1.4. Ý nghĩa khoa học và thực tiễn
- **Ý nghĩa khoa học:** Khóa luận bóc trần sự thật về hiện tượng báo cáo vượt mức (over-reporting) trong các bài báo khoa học liên quan đến dự đoán điểm số. Việc định nghĩa lại một giao thức kiểm chứng nghiêm ngặt giúp khôi phục lại tính minh bạch và chuẩn mực học thuật cho cộng đồng nghiên cứu EDM.
- **Ý nghĩa thực tiễn:** Mô hình CNN-BiLSTM được huấn luyện bằng giao thức Strict Validation đại diện cho khả năng dự báo thực tế khi hệ thống được triển khai tại các trường đại học, ngăn chặn tình trạng hệ thống AI bị "ảo giác" khi đối mặt với dữ liệu sinh viên hoàn toàn mới.

## 1.5. Cấu trúc của khóa luận
Khóa luận được cấu trúc thành 6 chương chính:
- **Chương 1: Mở đầu** - Giới thiệu bối cảnh, lý do chọn đề tài và mục tiêu.
- **Chương 2: Cơ sở lý thuyết** - Trình bày các kiến thức nền tảng về học sâu, CNN, BiLSTM và các hàm tối ưu.
- **Chương 3: Nghiên cứu liên quan** - Điểm qua các công trình trước đây và phân tích vấn đề rò rỉ dữ liệu.
- **Chương 4: Phương pháp đề xuất** - Trình bày chi tiết kiến trúc mạng Multi-Scale CNN-BiLSTM, chiến lược trích xuất đặc trưng và giao thức đánh giá.
- **Chương 5: Thực nghiệm và Đánh giá** - Trình bày kết quả so sánh, quá trình tối ưu hóa với Optuna, và phân tích chuyên sâu về phương sai dữ liệu.
- **Chương 6: Kết luận và Hướng phát triển** - Tổng kết đóng góp và đề xuất mở rộng.


# CHƯƠNG 2: CƠ SỞ LÝ THUYẾT

## 2.1. Khai phá dữ liệu giáo dục (Educational Data Mining - EDM)
Khai phá dữ liệu giáo dục (EDM) là một lĩnh vực nghiên cứu liên ngành tập trung vào việc áp dụng các phương pháp phân tích thống kê, khai phá dữ liệu và học máy để nhận diện các mô hình (patterns) ẩn giấu trong dữ liệu học tập. Dữ liệu trong EDM rất đa dạng, có thể là dữ liệu tĩnh như thông tin nhân khẩu học (giới tính, độ tuổi, nghề nghiệp phụ huynh), hoặc dữ liệu động như hành vi tương tác trên hệ thống e-learning (số lần nhấp chuột, thời gian làm bài, tỷ lệ nộp bài muộn).

Mục đích cốt lõi của EDM là "thấu hiểu" người học. Bằng việc phân tích lịch sử tương tác, các chuyên gia giáo dục có thể cá nhân hóa lộ trình học tập, tối ưu hóa giao diện phần mềm học trực tuyến, và đặc biệt là dự đoán kết quả học tập cuối kỳ. Khác với các hệ thống phân tích kinh doanh, dữ liệu EDM mang đậm tính nhận thức và hành vi con người, do đó nó thường chứa nhiều nhiễu (noise) và có độ biến thiên cao. Điều này đòi hỏi các thuật toán không chỉ có khả năng nội suy tốt mà còn phải có cơ chế trích xuất các đặc trưng hành vi phức tạp.

## 2.2. Mạng Nơ-ron Tích chập 1 Chiều (1D-CNN) áp dụng cho Dữ liệu Bảng
Mạng Nơ-ron Tích chập (Convolutional Neural Network - CNN) là một biến thể của mạng nơ-ron nhân tạo nhiều lớp, nổi tiếng với khả năng xử lý dữ liệu có cấu trúc lưới không gian (như hình ảnh 2D). Thành phần cấu trúc chính của mạng CNN bao gồm các lớp tích chập (Convolutional Layers) và các lớp tổng hợp (Pooling Layers).

Tuy nhiên, trong bài toán dự báo điểm số, dữ liệu đầu vào không phải là hình ảnh mà là các ma trận đặc trưng ở định dạng bảng (Tabular Data) với các dòng là sinh viên và các cột là thuộc tính. Để áp dụng CNN cho dạng dữ liệu này, chúng tôi sử dụng Mạng Tích chập Một chiều (1D-CNN). Một lớp 1D-CNN sẽ trượt một cửa sổ bộ lọc (filter/kernel) dọc theo vector đặc trưng của từng sinh viên để thực hiện phép toán tích chập. Phép toán này giúp mô hình tự động nhận diện và trích xuất các "cụm đặc trưng cục bộ" (local features). Ví dụ: mạng 1D-CNN có thể tự động học được rằng sự kết hợp giữa hai thuộc tính "thời gian vắng mặt cao" và "không có hỗ trợ học tập bổ sung" sẽ tạo ra một bộ lọc kích hoạt mạnh mẽ (high activation), đại diện cho dấu hiệu của việc kết quả học tập sa sút. Khả năng phát hiện tương quan cục bộ tự động này giúp 1D-CNN vượt trội hơn các mô hình mạng nơ-ron kết nối đầy đủ (Fully Connected Neural Networks - FCNN) truyền thống, vốn dễ bị quá khớp khi số lượng tham số quá lớn.

## 2.3. Mạng Bộ nhớ Ngắn hạn Hai chiều (BiLSTM)
Dữ liệu hành vi của học sinh không chỉ có các tính chất độc lập mà thường mang yếu tố thời gian hoặc sự liên kết dạng chuỗi (sequential). Chẳng hạn, điểm số thành phần (G1, G2) hay quỹ đạo tương tác của sinh viên tuân theo một tiến trình thời gian. Mạng Bộ nhớ Dài-Ngắn hạn (Long Short-Term Memory - LSTM) là một biến thể của mạng nơ-ron hồi quy (RNN) được thiết kế đặc biệt để giải quyết vấn đề biến mất gradient (vanishing gradient) khi học các chuỗi dữ liệu dài. LSTM duy trì một "trạng thái tế bào" (cell state) được điều khiển bởi ba cổng logic (Cổng quên, Cổng cập nhật, Cổng xuất) để quyết định thông tin nào trong quá khứ cần giữ lại và thông tin nào cần loại bỏ.

Tuy nhiên, LSTM tiêu chuẩn chỉ xử lý chuỗi thông tin theo một chiều (từ quá khứ đến hiện tại). Mạng Bộ nhớ Ngắn hạn Hai chiều (BiLSTM) là một sự mở rộng mạnh mẽ của LSTM, bao gồm hai lớp LSTM độc lập: một lớp xử lý dữ liệu theo chiều xuôi và một lớp xử lý theo chiều ngược. Đầu ra của BiLSTM tại mỗi bước là sự kết hợp (thường là phép ghép - concatenation) của cả hai hướng. Đặc điểm này cho phép mô hình nhìn nhận dữ liệu hành vi của học sinh không chỉ dựa vào những gì đã xảy ra trước đó, mà còn dưới góc độ tổng thể của toàn bộ chuỗi sự kiện trong học kỳ. Trong kiến trúc lai, BiLSTM tiếp nhận chuỗi đặc trưng cao cấp được trích xuất từ 1D-CNN và tiến hành mô hình hóa mối quan hệ phi tuyến sâu sắc giữa các đặc trưng đó.

## 2.4. Các Thách thức Kỹ thuật: Mất cân bằng lớp và Data Leakage
Khi xây dựng các mô hình EDM, hai thách thức toán học lớn nhất luôn hiện diện:
1. **Mất cân bằng lớp (Class Imbalance):** Trong môi trường giáo dục thực tế, phần lớn sinh viên có kết quả học tập ở mức trung bình và khá. Số lượng sinh viên xuất sắc (Điểm A) hoặc sinh viên rớt môn (Điểm F) chiếm tỷ lệ rất nhỏ. Nếu sử dụng các hàm mất mát truyền thống (như Cross-Entropy), mô hình học sâu sẽ bị chìm ngập trong lượng dữ liệu của lớp đa số và phớt lờ việc học cách nhận diện lớp thiểu số. Hậu quả là mô hình có độ chính xác tổng thể (Accuracy) cao, nhưng lại dự đoán sai hoàn toàn các sinh viên rớt môn.
2. **Rò rỉ dữ liệu (Data Leakage):** Xảy ra khi thông tin từ tập kiểm thử (Test set) vô tình lọt vào quá trình huấn luyện hoặc chọn lựa mô hình. Dữ liệu giáo dục thường có kích thước rất nhỏ (vài trăm sinh viên). Việc sử dụng một tập dữ liệu nhỏ để vừa kiểm tra (Validation) vừa báo cáo kết quả (Testing) sẽ gây ra sự lạc quan giả tạo. Mô hình bị quá khớp (Overfitted) vào sự xáo trộn ngẫu nhiên của tập kiểm thử thay vì học được bản chất quy luật.

## 2.5. Các Kỹ thuật Tối ưu Hóa Hiện Đại
Để vượt qua các thách thức trên, đề tài áp dụng các kỹ thuật tối ưu hóa hiện đại nhất trong lĩnh vực học sâu:
- **Focal Loss:** Là một hàm mất mát được điều chỉnh nhằm giải quyết bài toán mất cân bằng phân lớp cực độ. Bằng cách thêm một hệ số điều chế $(1 - p_t)^\gamma$ vào hàm Cross-Entropy, Focal Loss chủ động giảm trọng số của các mẫu dễ phân loại (nhóm điểm trung bình) và tập trung toàn bộ độ dốc (gradient) vào các mẫu khó (nhóm rớt môn hoặc xuất sắc). Tham số $\gamma$ (focusing parameter) kiểm soát mức độ tập trung này.
- **Mixup Regularization:** Là kỹ thuật tăng cường dữ liệu tiên tiến. Thay vì chỉ huấn luyện trên các mẫu dữ liệu rời rạc, Mixup nội suy tuyến tính giữa hai mẫu ngẫu nhiên và nhãn tương ứng của chúng. Điều này buộc mô hình phải học các quyết định phân lớp mượt mà (smooth decision boundaries), giúp mạng chống lại hiện tượng ghi nhớ học vẹt (memorization) và hoạt động bền bỉ hơn khi gặp dữ liệu nhiễu.
- **Tối ưu hóa bằng Optuna (Bayesian Optimization):** Optuna là một nền tảng tìm kiếm siêu tham số thông minh. Thay vì thử nghiệm mù quáng (Grid Search) hay ngẫu nhiên (Random Search), Optuna sử dụng các thuật toán dựa trên mô hình định hướng (Tree-structured Parzen Estimator - TPE). Optuna xây dựng một mô hình xác suất đánh giá hiệu năng dựa trên lịch sử các lần chạy trước đó, từ đó dự đoán và chỉ tập trung khám phá các khu vực tham số có triển vọng cao nhất (như Learning Rate, Dropout rate, Số lớp mạng). Điều này đặc biệt quan trọng đối với mạng lai CNN-BiLSTM vốn chứa hàng tá siêu tham số tương tác phức tạp.


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
- **Đặc trưng Đa thức (Polynomial Features):** Đối với các biến liên tục quan trọng như G1, G2, sự khác biệt giữa các mức điểm được làm rõ nét bằng cách tạo ra các biến bậc 2 ($G1^2, G2^2$) và biến tương tác ($G1 	imes G2$). Phép biến đổi phi tuyến này giúp mạng nơ-ron nhận diện tốt hơn những sự cải thiện hoặc sa sút học lực cực đoan.
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
