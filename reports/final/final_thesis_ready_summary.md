# Tóm tắt mô hình xAPI cuối cùng

Đối với bộ dữ liệu xAPI, mô hình deep learning cuối cùng được chọn là `gated_fusion_v28` với cơ chế dự đoán `low_f1_tuned`. Mô hình vẫn bám theo hướng CNN + BiLSTM của đề tài, kết hợp nhánh chuỗi hành vi học tập với nhánh ngữ cảnh thông qua gated fusion.

| Dataset | Mô hình | Chế độ dự đoán | Macro F1 | Recall Low | F1 Low |
|---|---|---|---:|---:|---:|
| xAPI | gated_fusion_v28 | low_f1_tuned | 0.7541 | 0.8846 | 0.8214 |

Baseline machine learning chỉ được dùng làm đối chứng cuối cùng, không được dùng để huấn luyện, distillation, pseudo-labeling, lấy xác suất dự đoán hoặc feature importance cho mô hình deep. Locked test chỉ dùng cho đánh giá cuối cùng; threshold được tinh chỉnh bằng xác suất CV/OOF. Không sử dụng ADASYN, không sử dụng student-combine và không claim regression head.
