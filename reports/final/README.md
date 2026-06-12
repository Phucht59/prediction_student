# Trạng thái kết quả

Các file có `_v27` trong tên là kết quả legacy của kiến trúc cũ. Không sử dụng các số liệu đó làm kết quả cuối của đề tài sau refactor.

Kết quả chính thức CNN-BiLSTM + MLP đã được sinh lại ngày 12/06/2026 với 50 trials cho mỗi dataset. Sử dụng các file không có `_v27` trong tên:

- `student-mat_3class_locked_test_metrics.json`
- `student-por_3class_locked_test_metrics.json`
- `xapi_3class_locked_test_metrics.json`

Các report chính thức ghi rõ:

- `Architecture: CNN-BiLSTM + Context MLP`
- `Loss: Weighted CrossEntropyLoss`
- prediction, confidence và xác suất từng lớp
- learning path theo giai đoạn
- bản ghi tương ứng trong PostgreSQL
