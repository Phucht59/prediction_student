# Xây dựng mô hình học kết hợp để dự đoán thành tích học tập sinh viên

Repository này là bản phát hành khóa luận cuối cùng: dữ liệu → tiền xử lý → mô hình
CNN-BiLSTM → dự đoán → hồ sơ rủi ro → khuyến nghị → PostgreSQL/bằng chứng.

## Final prediction models

| Model | Dataset | Macro-F1 | Balanced Accuracy | PR-AUC |
|---|---|---:|---:|---:|
| CNN-BiLSTM MAT | Student-Mat | **0.9015** | 0.9021 | 0.9442 |
| CNN-BiLSTM POR | Student-Por | **0.8623** | 0.8676 | 0.9147 |
| CNN-BiLSTM OULAD | OULAD | **0.8281** | 0.8203 | 0.8934 |

`cnn_bilstm_mat`, `cnn_bilstm_por` và `cnn_bilstm_oulad` là ba technical ID
duy nhất của model final. Đặc điểm transfer learning, temporal pretraining và
auxiliary objectives là cấu hình kỹ thuật, không phải tên model.

## Architecture

Student-Mat dùng CNN-BiLSTM với transfer learning, shared trunk và
subject-specific head được chọn hoàn toàn trên inner validation. Student-Por
dùng final CNN-BiLSTM probability ensemble.

CNN-BiLSTM OULAD nhận 47 temporal channels, qua input projection, multi-kernel
CNN, residual, BiLSTM và masked pooling; sau đó hợp nhất temporal, aggregate và
static branches bằng gated residual fusion. Risk là main head; survival và
outcome là auxiliary heads với trọng số cố định 0.15 cho mỗi head.

## Datasets and protocol

- Student-Mat và Student-Por: dự đoán ba lớp Low/Medium/High.
- OULAD: dự đoán rủi ro nhị phân tại cutoff F2_MIDDLE.
- Mọi lựa chọn model/hyperparameter dùng inner validation; outer test không
  được dùng để tuning.
- Future OULAD luôn ở trạng thái `LOCKED_NOT_EXECUTED`.
- Threshold OULAD cố định theo fold: 0.455, 0.495 và 0.500.

## Run validation

```powershell
python project.py final status
python project.py final report
python project.py final validate
pytest
```

Các lệnh final chỉ dựng lại báo cáo từ bằng chứng đã đóng băng, kiểm checksum,
checkpoint và replay; không train hoặc tune model.

## Teacher-feedback evidence

The final comparator catalog contains 10 models per dataset, including `MLP`
as one comparator. The fair UCI timing benchmark runs all 10 models on:

- `S0_EARLY_NO_GRADE`: context only, without G1/G2;
- `S1_MID_G1_ONLY`: context plus G1;
- `S2_LATE_G1_G2`: the frozen final UCI information contract.

The benchmark uses identical frozen outer rows and information availability,
training-only preprocessing, three inner folds, five fixed seeds, and no
Student-Por→Student-Mat transfer. `GRADE_BAND_REFERENCE` is reported as a
training-fold-only diagnostic and is not an eleventh model identity.

The generated evidence is stored in
`artifacts/final/uci_timing_scenarios/` and
`artifacts/final/teacher_feedback_validation/`. Training is explicit:

```powershell
python project.py study early-warning all
python project.py study early-warning validate
```

`python project.py final validate` remains read-only and never trains a model.
The OULAD audit confirms that 47 means channels per valid week (16 observed
weekly channels plus 31 current/past-only dynamics), with a separate padding
mask and no static/aggregate duplication. Recommendation expert evaluation
remains `PENDING_EXPERT_LABELS`.

## Repository structure

- `src/data`, `src/models`, `src/training`, `src/evaluation`: prediction stack.
- `src/recommendation`: risk profile, observed state, policy và validation.
- `src/database`, `database/final`: PostgreSQL architecture cho `system`,
  `catalog`, `ml` và `recommendation`.
- `configs/final`: bốn cấu hình public cuối.
- `artifacts/final`: checkpoint, prediction, tuning, ablation, calibration,
  recommendation, database và checksum evidence.
- `reports/final`: báo cáo bảo vệ và release audit.
- `tests`: contract, replay, leakage, recommendation và database validation.

Research history chỉ tồn tại cục bộ trong `test_lab/` (được Git ignore). Public
release không phụ thuộc vào thư mục này.

## Recommendation

Recommendation là deterministic decision-support, không phải Accuracy và không
phải tuyên bố causal effectiveness. Trong 15,378 records: 10,953 `GENERATED`,
1,209 `PARTIAL_EVIDENCE`, 3,216 `ABSTAINED`; generated-or-partial là 79.09% và
abstention là 20.91%. Plan `ABSTAINED` vẫn được lưu để truy vết nhưng có
`recommended_actions = []`.

## Evidence and database

Nguồn authority là
[`artifacts/final/final_results.csv`](artifacts/final/final_results.csv) và
[`reports/final/FINAL_MODEL_RESULTS.md`](reports/final/FINAL_MODEL_RESULTS.md).
Tuning trials, negative results, bootstrap, calibration, top-k, final
checkpoints, recommendation lineage và database audit đều nằm dưới
`artifacts/final`.

Database lưu 15,378 risk profiles, 15,378 plan objects và 27,355 actions.

## Scientific limitations

CNN-BiLSTM không được tuyên bố vượt trội phổ quát so với machine learning.
Engineered XGBoost đạt Macro-F1 xấp xỉ 0.828381 trong operational cross-check.
Prediction và recommendation không phải quan hệ nhân quả; hiệu quả can thiệp
cần đánh giá độc lập với chuyên gia/người dùng và outcome thực tế.
