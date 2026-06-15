# PHẦN KHUYẾN NGHỊ RA-HLPR (MỤC 3.5 & BẢNG KẾT QUẢ ĐÁNH GIÁ)

## 3.5. Hệ thống Khuyến nghị Lộ trình Học tập Hỗn hợp Thích ứng Rủi ro (RA-HLPR)
Hệ thống Khuyến nghị Lộ trình Học tập Hỗn hợp Thích ứng Rủi ro (Risk-Aware Hybrid Learning Path Recommender - RA-HLPR) hoạt động như một mô-đun hạ nguồn (downstream) độc lập, nhận đầu vào từ kết quả phân loại của mô hình chính và các đặc trưng của người học.

### 3.5.1. Đầu chẩn đoán rủi ro (Risk Diagnosis Head)
Đầu chẩn đoán rủi ro (Risk Diagnosis Head) là một mạng thần kinh MLP 3 lớp, nhận đầu vào là các đặc trưng của sinh viên kết hợp với phân phối xác suất dự đoán của mô hình phân loại chính. Thành phần này chẩn đoán các nguy cơ học thuật cụ thể dưới dạng các xác suất rủi ro. Số lượng đầu ra rủi ro được điều chỉnh tự động tùy thuộc vào bộ dữ liệu (6 rủi ro cho dữ liệu học sinh student-mat/por, và 3 rủi ro cho dữ liệu xapi).

### 3.5.2. Cơ sở tri thức can thiệp (Intervention Knowledge Base)
Cơ sở tri thức (Intervention Knowledge Base) lưu trữ các biện pháp can thiệp học thuật được chuẩn hóa trong file 'intervention_catalog.csv'. Mỗi can thiệp được định nghĩa bằng các thuộc tính như: mã can thiệp (item_id), tên biện pháp, mô tả chi tiết, nhóm rủi ro hướng tới (target_risks), độ khó sư phạm (difficulty_level), số giờ tự học ước tính hàng tuần (estimated_hours_per_week), giai đoạn đề xuất (recommended_phase), hiệu năng kỳ vọng (expected_effect) và yêu cầu kiến thức tiên quyết (prerequisite_level).

### 3.5.3. Chiến lược gán nhãn yếu (Weak Labeling Strategy)
Do dữ liệu thực tế không có sẵn nhãn rủi ro cụ thể của từng học sinh, phương pháp gán nhãn yếu (Weak Labeling) dựa trên tri thức chuyên gia được áp dụng để sinh nhãn huấn luyện cho đầu chẩn đoán rủi ro. Các quy tắc gán nhãn yếu được thiết kế chặt chẽ theo nguyên tắc 'Không dùng risk không có feature' nhằm tránh thiên kiến học máy. Cụ thể, đối với dữ liệu student, toàn bộ 6 rủi ro được ánh xạ thông qua các thuộc tính hiện có (failures, G1/G2, absences, freetime/goout, studytime). Đối với dữ liệu xapi, các rủi ro không có đặc trưng tương ứng (như lịch sử trượt môn, điểm số lịch sử, thời gian tự học) sẽ được loại bỏ, chỉ thực hiện gán nhãn yếu cho 3 rủi ro có dữ liệu hỗ trợ (nghỉ học, mức độ tương tác LMS, rủi ro học lực yếu). Việc huấn luyện đầu chẩn đoán được thực hiện bằng cách sử dụng hàm lỗi BCEWithLogitsLoss có trọng số pos_weight để cân bằng nhãn.

### 3.5.4. Bộ chấm điểm hỗn hợp (Hybrid Scorer) và Bộ lọc ứng viên (Candidate Generator)
Bộ chấm điểm hỗn hợp (Hybrid Scorer) tính điểm ưu tiên cho từng biện pháp can thiệp dựa trên công thức đa tiêu chí tối ưu: score = 0.3 * risk_match + 0.2 * performance_need + 0.15 * difficulty_fit + 0.15 * time_fit + 0.1 * prerequisite_fit + 0.1 * expected_effect. Trước khi chấm điểm, Bộ lọc ứng viên (Candidate Generator) sẽ lọc bớt các can thiệp không phù hợp với mức độ rủi ro hiện tại (xác suất rủi ro hướng tới phải từ 0.3 trở lên) và lớp học lực dự đoán để tối ưu hóa hiệu suất tính toán và tăng độ tập trung sư phạm.

### 3.5.5. Bộ lập lộ trình học tập (Learning Path Planner)
Bộ lập lộ trình học tập (Learning Path Planner) phân bổ các biện pháp can thiệp đã được chấm điểm vào một lộ trình 4 tuần tuần tự theo các chủ đề sư phạm tăng tiến: Tuần 1: Ổn định (Stabilize - giải quyết rào cản khẩn cấp), Tuần 2: Thực hành (Practice - bù đắp hổng kiến thức), Tuần 3: Củng cố (Reinforce - tăng tương tác học tập), Tuần 4: Đánh giá & Điều chỉnh (Evaluate & Adjust - đánh giá lại hoặc thách thức nâng cao). Đồng thời, hệ thống tự động sinh ra các diễn giải tiếng Việt thân thiện giải thích lý do cụ thể đề xuất các can thiệp này dựa trên hồ sơ của từng học sinh.

### * Hạn chế của phương pháp đánh giá
Mặc dù các chỉ số đo lường lộ trình học tập (độ phủ rủi ro, tính tăng tiến khó dần, độ vi phạm tiên quyết, độ cân bằng tải) đều đạt kết quả tốt trên dữ liệu mô phỏng, phương pháp này vẫn tồn tại hạn chế lớn là thiếu kiểm chứng thực nghiệm thực tế (longitudinal validation/A-B Testing) trên người học thực tế trong thời gian dài để chứng minh hiệu quả nâng cao kết quả học tập cuối cùng.

---

## BẢNG 4.1: KẾT QUẢ CHẨN ĐOÁN RỦI RO VÀ XẾP HẠNG CAN THIỆP CỦA RA-HLPR

| Bộ dữ liệu | Micro F1 | Macro F1 | Precision@3 | NDCG@3 | Catalog Coverage |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **student-mat** | 0,9593 | 0,9613 | 0,8650 | 0,8830 | 1,0000 |
| **student-por** | 0,9311 | 0,9399 | 0,8410 | 0,8629 | 1,0000 |
| **xapi** | 0,9765 | 0,6543 | 0,7708 | 0,9081 | 0,8333 |

---

## BẢNG 4.2: ĐÁNH GIÁ CHẤT LƯỢNG LỘ TRÌNH HỌC TẬP 4 TUẦN

| Bộ dữ liệu | Độ phủ Rủi ro | Độ khó Tăng tiến | Vi phạm Tiên quyết | Cân bằng Tải học tập |
| :--- | :---: | :---: | :---: | :---: |
| **student-mat** | 0,8707 | 0,4852 | 0,0401 | 1,8242 |
| **student-por** | 0,9176 | 0,4564 | 0,0167 | 1,7648 |
| **xapi** | 1,0000 | 0,6528 | 0,0052 | 2,0100 |
