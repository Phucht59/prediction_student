# Kiến trúc hệ thống cuối

Repository công khai một họ mô hình chính: CNN-BiLSTM, với ba định danh
`cnn_bilstm_mat`, `cnn_bilstm_por` và `cnn_bilstm_oulad`.

```text
raw data
  -> training-only preprocessing
  -> temporal/context tensors + availability mask
  -> CNN-BiLSTM shared representation
  -> class/risk probability
  -> calibration and evidence
  -> risk profile
  -> constrained recommendation
  -> PostgreSQL
```

UCI dùng một estimator cho S0/S1/S2; OULAD dùng một estimator cho
E1/E2/M1/L1. Checkpoint không mang stage identity. Mask/cutoff kiểm soát lượng
thông tin có thể quan sát ở từng thời điểm.

`src/pipelines/` chứa pipeline dự đoán; `src/models/` chứa kiến trúc;
`src/final_release/` chỉ tổng hợp/kiểm tra evidence đóng băng và không có entry
point train ngầm. `src/studies/` chỉ còn hai shim nhỏ để load checkpoint joblib
đã đóng băng với import path cũ.

Chi tiết kiến trúc, protocol và kết quả được trình bày trong `PROJECT.md`.
