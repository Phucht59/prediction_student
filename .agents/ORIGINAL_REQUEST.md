# Original User Request

## Initial Request — 2026-06-14T05:15:18Z

# Teamwork Project Prompt

Khôi phục và xây dựng lại Mô hình Khuyến nghị bằng Học máy (ML/DL) cùng hệ thống Đánh giá khoa học, nhưng **TUYỆT ĐỐI GIỮ NGUYÊN** logic tiền xử lý dữ liệu và thuật toán Resampling ban đầu của hệ thống.

Working directory: c:\Huflit\kltn
Integrity mode: development

## Requirements

### R1. Xây dựng Mô hình Khuyến nghị bằng ML/DL (PyTorch MLP)
Thay thế `RuleBasedLearningPathEngine` hiện tại bằng một mô hình mạng nơ-ron (Multi-Layer Perceptron) để dự đoán và sinh lộ trình học tập cá nhân hóa. Đội ngũ AI tự do thiết kế kiến trúc mạng phù hợp với dữ liệu (`student-mat`, `student-por`, `xapi`).

### R2. Xây dựng Hệ thống Đánh giá Khoa học (Evaluation Pipeline)
Xây dựng một script riêng biệt (như `src/eval_recommendation.py`) để đánh giá lộ trình khuyến nghị:
1. **Định lượng:** Tính toán các chỉ số xếp hạng (Precision@K, Recall@K, NDCG@K).
2. **Định tính:** Tích hợp bộ chấm điểm tự động LLM-Judge theo tiêu chí sư phạm.
Lưu kết quả đầu ra thành các file JSON vào thư mục `reports/final/recommendations/`.

### R3. Ràng buộc Kỹ thuật Nghiêm ngặt (Constraint)
**TUYỆT ĐỐI KHÔNG ĐƯỢC CHẠM VÀO** hoặc thay đổi bất kỳ dòng code nào liên quan đến kỹ thuật Resampling (ADASYN/SMOTENC), ép kiểu dữ liệu (casting), hoặc Data Preprocessing trong các file `src/data_pipeline.py` hay `src/train_pipeline.py`. Giữ nguyên kết quả gốc vì nó đang mang lại hiệu năng cao nhất.

## Acceptance Criteria

### Tính chính xác và Logic
- [ ] Codebase sử dụng mạng PyTorch MLP cho phần Learning Path thay vì rule-based if-else.
- [ ] Logic của thuật toán ADASYN/SMOTE vẫn giữ nguyên bản như cũ, không có thêm các bước ép kiểu hay cắt xén dữ liệu.

### Đánh giá và Kiểm định
- [ ] Có script `src/eval_recommendation.py` chạy độc lập sinh ra file JSON chứa NDCG, Precision, và điểm LLM-Judge.
- [ ] Toàn bộ hệ thống chạy end-to-end không phát sinh lỗi.

## Follow-up — 2026-06-14T08:42:17Z

# Teamwork Project Prompt — Report Update

Cập nhật mã nguồn sinh Báo cáo Khóa luận (`generate_doc.py`) để tạo ra bản báo cáo mới nhất, phản ánh chính xác 100% mô hình Khuyến nghị ML/DL mới vừa được xây dựng, đồng thời xóa bỏ hoàn toàn các nhắc nhớ về tập luật cũ. 

**Lưu ý cực kỳ quan trọng:** TUYỆT ĐỐI Không nhắc đến việc sửa lỗi thuật toán Resampling vì người dùng đã yêu cầu giữ nguyên phương pháp Resampling gốc (ADASYN/SMOTE bị lỗi ép kiểu) do nó đem lại F1 tốt hơn.

Working directory: c:\Huflit\kltn
Integrity mode: development

## Requirements

### R1. Cập nhật Nội dung Báo cáo (Chương 3)
Cập nhật nội dung trong file Python sinh báo cáo để phản ánh kiến trúc hệ thống hiện tại:
- **Mô hình Khuyến nghị:** Xóa bỏ hoàn toàn mô tả về tập luật if-else (Rule-based). Viết lại chi tiết kiến trúc mạng nơ-ron đa tầng (PyTorch MLP) hiện đại đã được dùng để dự đoán và sinh lộ trình học tập.

### R2. Cập nhật Kết quả Đánh giá (Chương 4 & 5)
- Bổ sung bảng biểu/kết quả của hệ thống Đánh giá Mô hình Khuyến nghị dựa trên các file báo cáo JSON (bao gồm các chỉ số khoa học: Precision@K, Recall@K, NDCG@K và đánh giá LLM-Judge) để giải quyết triệt để nhận xét của giảng viên. Thư mục chứa JSON là `reports/final/recommendations`.
- Cập nhật mục "Hạn chế của đề tài" thành "Những vấn đề đã khắc phục" (Giải thích việc chuyển từ Rule-based sang ML/DL để tăng tính cá nhân hóa) và định hướng "Hướng phát triển tương lai".

### R3. Hoàn thiện Format Báo cáo
Đảm bảo script sinh ra file Word (.docx) chuẩn chỉnh theo format hiện tại, văn phong khoa học, logic chặt chẽ, đáp ứng toàn bộ các góp ý trong file PDF nhận xét. Hệ thống cần chạy thử file script để đảm bảo ra được thành phẩm `Bao_cao_cuoi_cung.docx` (bạn hãy sửa code để output ra file này thay vì Bao_cao_tien_do.docx).

## Acceptance Criteria

### Tính chính xác của nội dung
- [ ] File Word được sinh ra KHÔNG còn bất kỳ chữ "Rule-based" nào cho phần Learning Path.
- [ ] File Word CÓ mục trình bày lý thuyết và kiến trúc mạng PyTorch MLP cho hệ thống khuyến nghị.
- [ ] File Word CÓ mục kết quả đánh giá bằng chỉ số Ranking (NDCG, Precision) và LLM-Judge rõ ràng, đầy đủ cơ sở khoa học.

### Khả năng thực thi
- [ ] Script `generate_doc.py` chạy thành công hoàn toàn, tự động load các thông số từ file JSON kết quả để điền vào Word.
- [ ] Thành phẩm cuối cùng là một bản Word `Bao_cao_cuoi_cung.docx` hoàn chỉnh, đọc liền mạch và mang đậm chất báo cáo học thuật.
