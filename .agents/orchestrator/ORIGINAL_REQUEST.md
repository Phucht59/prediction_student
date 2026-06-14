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

## Follow-up — 2026-06-14T15:24:15+07:00

The previous orchestrator run was interrupted by a resource limit (quota error). The quota has now reset.
Please examine the current workspace, read c:\Huflit\kltn\.agents\ORIGINAL_REQUEST.md, inspect the files in your directory (plan.md, context.md, progress.md), and resume work to complete the project.
Specifically:
1. Rebuild the Recommendation Model using a PyTorch MLP.
2. Build a scientific Evaluation Pipeline in src/eval_recommendation.py saving results to reports/final/recommendations/.
3. Ensure no changes are made to preprocessing or resampling logic in src/data_pipeline.py or src/train_pipeline.py.

Communicate progress back to me by updating progress.md frequently. When the project is complete, notify me with a summary of the accomplishments.
