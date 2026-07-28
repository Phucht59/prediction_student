# Xây dựng mô hình học kết hợp để dự đoán thành tích học tập sinh viên

Repository này là source of truth của dự án khóa luận: dữ liệu → tiền xử lý →
dự đoán theo giai đoạn → hồ sơ rủi ro → khuyến nghị → PostgreSQL/bằng chứng.

## Hệ thống dự đoán cuối

UCI dùng một estimator cho mỗi dataset/model/fold/seed và estimator đó dự đoán
cả ba thời điểm:

- `S0_EARLY_NO_GRADE`: trước G1, không dùng G1/G2;
- `S1_MID_G1_ONLY`: sau G1, không dùng G2;
- `S2_LATE_G1_G2`: sau G2, trước G3.

Stage là chiều của prediction, không phải model identity. Vì vậy UCI có 10 model
families × 2 datasets = 20 identities và 60 stage-result rows, không phải 60
model khác nhau. Mỗi base record giữ nguyên outer fold ở cả ba stage.

CNN-BiLSTM dùng temporal tensor `[2, 7]`, availability mask và context branch.
S0 không có temporal timestep; S1 chỉ có đúng một recurrent timestep hợp lệ;
S2 dùng đủ hai timestep. Một shared classification head được huấn luyện bằng
trung bình cross-entropy cân bằng giữa S0/S1/S2.

## Frozen thesis evidence

Các artifact chính thức trước refactor không bị ghi đè:

| Model | Dataset | Frozen Macro-F1 |
|---|---|---:|
| CNN-BiLSTM MAT | Student-Mat | 0.9014601961315334 |
| CNN-BiLSTM POR | Student-Por | 0.8622587167738002 |
| CNN-BiLSTM OULAD | OULAD | 0.8280835945631038 |

Technical IDs vẫn là `cnn_bilstm_mat`, `cnn_bilstm_por` và
`cnn_bilstm_oulad`. Các số trên là regression guard cho bằng chứng khóa luận đã
đóng băng; authority vận hành mới theo stage nằm trong
`artifacts/final/final_stage_results.csv` và
`artifacts/final/final_overall_results.csv`.

## OULAD

OULAD giữ nguyên model và checkpoint tại stage `F2_MIDDLE`. Không có OULAD
retraining trong refactor này. Future OULAD luôn là
`LOCKED_NOT_EXECUTED`.

## Protocol

- Target UCI: Low (`G3 < 10`), Medium (`10 <= G3 < 15`), High (`G3 >= 15`).
- G3 chỉ tạo target, không đi vào predictor.
- Frozen outer folds; ba inner folds.
- Seeds cố định: 42, 1201, 2026, 3407, 7319.
- Preprocessing fit training-only.
- Không outer tuning, best-seed selection, transfer, pretrained checkpoint,
  SMOTE/ADASYN hoặc synthetic tensor resampling.
- Grade-band reference chỉ là diagnostic training-fold-only, không phải model
  thứ 11 và không được fusion vào model.

## Commands

```powershell
python project.py study unified-stage prepare
python project.py study unified-stage train
python project.py study unified-stage evaluate
python project.py study unified-stage report
python project.py study unified-stage validate

python project.py final status
python project.py final report
python project.py final validate
pytest
ruff check .
```

`final validate` chỉ đọc/kiểm tra evidence; training luôn là command explicit.
Resume key của unified training là dataset/model/fold/seed/config hash, không có
scenario.

## Evidence

- `artifacts/final/unified_stage_aware_uci/`: protocol, split, checkpoints,
  predictions, metrics, bootstrap và checksum.
- `reports/final/UNIFIED_STAGE_AWARE_RESULTS.md`: kết quả theo stage.
- `reports/final/HYBRID_VS_ML_STAGE_MATRIX.md`: CNN-BiLSTM so với comparator.
- `reports/final/UNIFIED_MODEL_SELECTION_REPORT.md`: inner selection.
- `artifacts/history/legacy_uci_separate_stage_v1/`: evidence separate-stage cũ,
  giữ nguyên checksum để truy vết, không còn là authority.

## Database và recommendation

Schema replacement hỗ trợ `prediction_stage`, một run có nhiều stage
predictions/metrics, với prediction key `(run_id, record_pk, prediction_stage)`.
API có dạng `predict(record, stage)` và `recommend(record, stage)`.

Canonical database không bị cutover trong branch này. Một replacement database
riêng được dựng và kiểm tra trước, gồm unified UCI và OULAD không đổi.
Recommendation giữ nguyên 15.378 risk profiles, 15.378 plans và 27.355 actions;
expert evaluation vẫn `PENDING_EXPERT_LABELS`.

## Scientific limitations

Kết quả theo stage đo predictive association, không chứng minh quan hệ nhân quả
hay hiệu quả can thiệp. CNN-BiLSTM không được tuyên bố vượt trội phổ quát so với
machine learning. S0/S1 có ít thông tin hơn S2; so sánh phải luôn giữ đúng
information contract. xAPI không thuộc final project.
