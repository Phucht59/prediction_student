# Bằng chứng xử lý mất cân bằng trên frozen Hybrid embeddings

## Mục tiêu

Thí nghiệm kiểm tra ảnh hưởng của mất cân bằng lớp mà **không huấn luyện lại và không thay thế checkpoint Hybrid CNN–BiLSTM chính thức**. Biểu diễn 96 chiều từ Hybrid được đóng băng; chỉ một Logistic Regression head giống nhau được huấn luyện dưới bốn chế độ:

- `none`: không tái cân bằng;
- `class_weight`: gán trọng số lớp;
- `SMOTE`: sinh mẫu tổng hợp chỉ trên train embeddings;
- `ADASYN`: sinh mẫu tổng hợp chỉ trên train embeddings.

Validation chỉ dùng chọn threshold. Test chỉ dùng báo cáo metric cuối. Validation và test không bao giờ được resample.

## Kết quả tổng thể

| Mode | ROC-AUC | PR-AUC | Precision | Recall | F1 | Balanced Accuracy | Specificity | Brier |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| none | 0.8438 | 0.8214 | 0.7437 | **0.6745** | **0.7074** | **0.7587** | 0.8428 | **0.1672** |
| class_weight | 0.8446 | 0.8219 | 0.7902 | 0.6259 | 0.6985 | 0.7568 | 0.8876 | 0.1841 |
| SMOTE | 0.8457 | 0.8218 | 0.8722 | 0.5175 | 0.6496 | 0.7331 | 0.9487 | 0.1909 |
| ADASYN | **0.8473** | **0.8233** | **0.8982** | 0.4788 | 0.6246 | 0.7211 | **0.9633** | 0.1814 |

## Diễn giải

- `none` cho F1, recall, balanced accuracy và Brier tốt nhất trong bảng tổng thể; đây là cấu hình cân bằng nhất nếu mục tiêu là không bỏ sót quá nhiều sinh viên nguy cơ.
- `class_weight` tăng precision và specificity nhưng làm recall giảm.
- `SMOTE` và `ADASYN` tăng precision mạnh, đồng thời giảm recall rõ rệt. Synthetic oversampling không tự động cải thiện mục tiêu cảnh báo sớm.
- ADASYN có ROC-AUC/PR-AUC cao nhất nhưng threshold cuối tạo trade-off quá nghiêng về precision; do đó không được tự động dùng để thay checkpoint authority.

## Kết quả theo stage

- `EARLY_20`: linear head trên frozen embeddings chưa phân biệt tốt hai lớp ở threshold được chọn; balanced accuracy bằng 0.50 và specificity bằng 0.
- `EARLY_35`: `class_weight` cho trade-off hợp lý với balanced accuracy 0.7149, precision 0.6094 và recall 0.7870.
- `MIDDLE_50`: `none` giữ F1 cao nhất; các chế độ synthetic tăng precision nhưng làm recall giảm mạnh.
- `LATE_75`: `class_weight` đạt F1 0.7666, balanced accuracy 0.8112; ADASYN tăng recall lên 0.8268 nhưng precision giảm còn 0.7463.

## Kết luận authority

Thí nghiệm này là **sensitivity evidence**, không phải model-selection authority. Checkpoint Hybrid chính thức không bị thay thế.

> Ảnh hưởng của mất cân bằng đã được kiểm tra trên frozen Hybrid embeddings bằng none, class weight, SMOTE và ADASYN. Synthetic oversampling chỉ được áp dụng trên train embeddings. Không có phương pháp resampling nào cải thiện đồng thời precision, recall, F1, balanced accuracy và calibration; vì vậy checkpoint Hybrid chính thức được giữ nguyên.

## Artefact

- `artifacts/recommend_hybrid/imbalance/FROZEN_IMBALANCE_EVIDENCE.json`
- `configs/recommend_hybrid/frozen_imbalance_evidence.yaml`
- `src/recommend_hybrid/imbalance.py`
- `scripts/recommend_hybrid/run_frozen_imbalance_evidence.py`
- `tests/recommend_hybrid/test_frozen_hybrid_imbalance.py`
