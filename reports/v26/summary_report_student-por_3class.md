# BÁO CÁO TỔNG KẾT V26 SCIENTIFIC HYBRID

- **Dataset**: student-por
- **Chế độ mục tiêu (Target Mode)**: 3class

## 1. Mục tiêu và Quy trình No-Leakage
Mô hình HybridCNNBiLSTMAttentionOrdinalV2 được huấn luyện với kỷ luật khắt khe:
- Trích lập 20% Locked Test ngay từ đầu.
- Mọi quá trình Scaler, LabelEncoding, Oversampling (SMOTE/ADASYN) CHỈ thực hiện nội bộ trong Train Pool (80%).
- Ensemble từ 5 mô hình huấn luyện với Fixed Seeds để giảm nhiễu hoàn toàn.

## 2. Kết quả Đánh giá (Locked Test Final)
- **Accuracy**: 0.8462
- **F1 Macro**: 0.8045
- **Precision Macro**: 0.8279
- **Recall Macro**: 0.7875

## 3. Siêu tham số tối ưu (Optuna Best Params)
- `learning_rate`: 0.0023554813120282058
- `weight_decay`: 1.4812870629350145e-05
- `dropout`: 0.40848200885281966
- `lstm_hidden`: 64
- `conv_channels`: 64
- `mlp_hidden`: 64
- `batch_size`: 32
- `lambda_ordinal`: 0.44745830082152505
- `focal_gamma`: 1.397923073617411
- `oversample_method`: smote

## 4. Hệ thống khuyến nghị (Recommendation Engine)
Đã tự động sinh khuyến nghị cá nhân hóa cho từng sinh viên trong tệp output `recommendations/`.
Chiến lược dựa trên luật kết hợp confidence score và các biến số risk rủi ro như (absence_study_ratio, grade_growth).

## 5. Hạn chế & Hướng cải tiến
- Kích thước dữ liệu gốc khá nhỏ (vài trăm samples), do đó dễ gặp overfit nếu không tuning hyperparameter cẩn thận.
- Việc bảo vệ tuyệt đối Locked Test có thể khiến metrics thấp hơn các paper cũ có áp dụng rò rỉ SMOTE, nhưng đây là kết quả ĐÁNG TIN CẬY và CHUẨN KHOA HỌC nhất.
