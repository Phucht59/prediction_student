# Scope: Report Update

## Architecture
- `generate_doc.py`: Script sinh tự động báo cáo Word sử dụng thư viện `docx` (python-docx).
- `reports/final/recommendations/*_evaluation.json`: Chứa các kết quả đánh giá mô hình MLP Recommendation bao gồm các chỉ số NDCG@K, Precision@K, Recall@K và thông tin LLM-Judge.

## Milestones
| # | Name | Scope | Dependencies | Status |
|---|------|-------|-------------|--------|
| 1 | Exploration | Khảo sát cấu trúc generate_doc.py và các tệp JSON | None | DONE |
| 2 | Implementation | Cập nhật mã nguồn generate_doc.py | M1 | IN_PROGRESS |
| 3 | Verification | Chạy thử nghiệm và kiểm chứng nội dung Bao_cao_cuoi_cung.docx | M2 | PLANNED |
| 4 | Audit | Chạy kiểm tra Forensic Auditor độc lập | M3 | PLANNED |

## Interface Contracts
- Tệp sinh ra phải có tên `Bao_cao_cuoi_cung.docx` và nằm ở thư mục gốc `c:\Huflit\kltn`.
- Không chứa từ khóa "Rule-based" cho Learning Path.
- Có mục mô tả kiến trúc mạng nơ-ron MLP phục vụ khuyến nghị học tập.
- Có bảng hiển thị kết quả đánh giá (metrics ranking và LLM-Judge) được load tự động từ tệp JSON.
- Không nhắc đến việc sửa lỗi Resampling (SMOTE/ADASYN).
