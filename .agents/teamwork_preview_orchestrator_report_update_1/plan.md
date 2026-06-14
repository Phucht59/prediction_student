# Orchestration Plan - Report Update Milestone

## Objective
Cập nhật mã nguồn sinh Báo cáo Khóa luận (`generate_doc.py`) để sinh ra `Bao_cao_cuoi_cung.docx`.
Bản báo cáo phải phản ánh mô hình Khuyến nghị ML/DL mới (PyTorch MLP), không đề cập tới "Rule-based" cho Learning Path, thêm phần mô tả lý thuyết/kiến trúc PyTorch MLP, hiển thị bảng số liệu đánh giá (ranking metrics NDCG/Precision và LLM-Judge) tự động tải từ các tệp JSON trong `reports/final/recommendations`, và TUYỆT ĐỐI không nhắc tới việc sửa lỗi thuật toán Resampling.

## Steps

### Step 1: Exploration
- Phân tích mã nguồn `generate_doc.py` hiện tại.
- Phân tích nội dung và cấu trúc của 3 tệp JSON kết quả đánh giá mô hình khuyến nghị:
  - `reports/final/recommendations/student_mat_evaluation.json`
  - `reports/final/recommendations/student_por_evaluation.json`
  - `reports/final/recommendations/xapi_evaluation.json`
- Lên cấu trúc bổ sung cho Chương 3 (Kiến trúc mô hình MLP) và Chương 4 (Kết quả đánh giá hệ khuyến nghị).
- Soạn thảo nội dung mô tả lý thuyết mạng nơ-ron MLP phục vụ xếp hạng rủi ro học tập.

### Step 2: Implementation
- Cập nhật mã nguồn `generate_doc.py`:
  - Đọc tự động các tệp JSON.
  - Sửa đổi đường dẫn lưu tệp đầu ra thành `Bao_cao_cuoi_cung.docx`.
  - Thay thế hoặc cập nhật phần Khuyến nghị học tập (Chương 3.5), đưa vào phần lý thuyết và kiến trúc của PyTorch MLP.
  - Thêm phần kết quả đánh giá hệ khuyến nghị (Chương 4) hiển thị bảng số liệu ranking metrics (Precision@K, Recall@K, NDCG@K cho K = 1, 3, 5) và kết quả LLM-Judge.
  - Kiểm tra loại bỏ từ khóa "Rule-based" hoặc "tập luật" liên quan tới Learning Path.
  - Đảm bảo TUYỆT ĐỐI không đề cập tới việc sửa lỗi Resampling (SMOTE/ADASYN).

### Step 3: Verification
- Chạy thử nghiệm `generate_doc.py` để sinh ra `Bao_cao_cuoi_cung.docx`.
- Kiểm tra sự tồn tại và cấu trúc nội dung của tệp `Bao_cao_cuoi_cung.docx` (sử dụng python-docx hoặc công cụ thích hợp để đọc văn bản và đối chiếu các từ khóa cấm/bắt buộc).

### Step 4: Forensic Audit
- Chạy Forensic Auditor để đánh giá tính toàn vẹn và đảm bảo không vi phạm các quy tắc đạo đức/kỹ thuật.
