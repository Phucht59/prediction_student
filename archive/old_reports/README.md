# Trạng thái kết quả

Kết quả chính thức hiện tại thuộc phiên bản `final model v1`. Không sử dụng các artifact hậu tố `_v27` hoặc kết quả của những lần chạy lại với cấu hình loss/resampling khác.

| Dataset | Optuna best CV F1 | Locked-test F1-Macro |
|---|---:|---:|
| Student-Mat | 0.9035 | 0.8690 |
| Student-Por | 0.8804 | 0.8156 |
| xAPI | 0.8233 | 0.7850 |

Nguồn locked-test chính thức là các file `reports/final/metrics/*_3class_locked_test_metrics.json`. Điểm CV được lưu riêng trong `*_3class_optuna_cv.json` để không trộn với phép đánh giá cấu hình cố định.

Mô hình dự đoán sử dụng CNN-BiLSTM + Context MLP và ensemble 11 seed. Student-Mat/Student-Por dùng ADASYN; xAPI dùng SMOTENC. Mô-đun khuyến nghị MLP được đánh giá riêng bằng Precision@K, Recall@K và NDCG@K.
