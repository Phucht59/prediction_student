# Dự đoán thành tích học tập sinh viên

Dự án xây dựng và đánh giá hệ thống dự đoán kết quả học tập ba mức trên bộ dữ
liệu UCI Student Performance (`student-mat`). Mô hình nghiên cứu cuối cùng là
CNN–BiLSTM nhận chuỗi hai điểm đánh giá trước đó `[G1, G2]`; hệ thống đồng thời
cung cấp benchmark Machine Learning, lưu vết dữ liệu bằng PostgreSQL và sinh
khuyến nghị hỗ trợ học tập theo luật xác định.

> Phạm vi khoa học: dữ liệu gồm 395 học sinh trung học tại Bồ Đào Nha trong môn
> Toán. Đây không phải dữ liệu sinh viên đại học Việt Nam. Kết quả chỉ chứng minh
> tính khả thi trên bộ dữ liệu nghiên cứu và cần được thẩm định lại trước khi áp
> dụng trong bối cảnh khác.

## Bài toán

Điểm cuối kỳ `G3` được chuyển thành ba lớp có thứ tự:

| Lớp | Điều kiện | Số mẫu toàn bộ |
| --- | ---: | ---: |
| Low | `G3 <= 9` | 130 |
| Medium | `10 <= G3 <= 14` | 192 |
| High | `G3 >= 15` | 73 |

Tập dữ liệu được chia cố định và phân tầng thành 316 mẫu development và 79 mẫu
locked test. `G3` chỉ được dùng làm nhãn, không bao giờ được đưa vào đặc trưng.

Ba thời điểm dự báo được định nghĩa riêng:

- `late_stage`: dùng G1 và G2;
- `early_warning`: loại G2;
- `pre_assessment`: loại cả G1 và G2.

Các thời điểm này không được so sánh trực tiếp như cùng một bài toán vì lượng
thông tin đầu vào khác nhau.

## Mô hình cuối cùng

Mô hình nghiên cứu được đóng băng là CNN–BiLSTM một seed (`42`):

```text
[G1, G2]
   -> Conv1D(1 -> 16, kernel=1)
   -> BatchNorm + ReLU + sequence dropout
   -> BiLSTM(hidden=32, 1 layer)
   -> concatenated bidirectional hidden state
   -> head dropout
   -> Linear(64 -> 3)
   -> Softmax / argmax
```

Mô hình có 13.059 tham số học được. Cấu hình cuối dùng batch size 32, learning
rate `0.0046677`, weight decay `0.0003541`, tối đa 40 epoch và early stopping
patience 12. Nhãn lịch sử `weighted_ce` trong cấu hình được diễn giải là
`CrossEntropyLoss` không class weighting vì `class_weight_mode=none`.

## Kết quả CNN–BiLSTM đã đóng băng

| Chỉ số | Kết quả |
| --- | ---: |
| Nested outer Macro-F1 | `0.8781 ± 0.0448` |
| Locked-test accuracy | `0.9114` |
| Locked-test Macro-F1 | `0.9262` |
| Locked-test weighted F1 | `0.9122` |
| Balanced accuracy | `0.9345` |
| Quadratic weighted kappa | `0.9152` |
| Macro PR-AUC | `0.9699` |
| Brier / ECE | `0.1683 / 0.0591` |
| Ordinal MAE | `0.0886` |

Khoảng tin cậy bootstrap 95% trên 79 mẫu locked test là `0.8481–0.9620` cho
accuracy và `0.8704–0.9694` cho Macro-F1. Có 7 lỗi lệch một mức và không có lỗi
lệch hai mức.

## Benchmark mô hình công bằng

Runner `scripts/run_fair_model_comparison.py` so sánh các mô hình sau bằng cùng
G1/G2, cùng 316 mẫu development, cùng 5 outer folds, 3 inner folds, 30 Optuna
trials cho mỗi mô hình/fold, seed 42, không resampling, không class weighting và
không sử dụng locked test:

- Decision Tree;
- Random Forest;
- SVM-RBF;
- XGBoost;
- Gradient Boosting;
- CNN+LSTM;
- CNN+BiLSTM.

| Mô hình | Outer Macro-F1 | OOF accuracy |
| --- | ---: | ---: |
| Random Forest | **`0.8915 ± 0.0240`** | `0.8861` |
| Decision Tree | `0.8906 ± 0.0248` | `0.8861` |
| SVM-RBF | `0.8894 ± 0.0290` | `0.8829` |
| Gradient Boosting | `0.8872 ± 0.0290` | `0.8829` |
| XGBoost | `0.8739 ± 0.0341` | `0.8703` |
| CNN+BiLSTM | `0.8380 ± 0.0475` | `0.8354` |
| CNN+LSTM | `0.7970 ± 0.1253` | `0.8133` |

Đây là benchmark kiến trúc riêng, không thay thế kết quả của cấu hình CNN–BiLSTM
final. CNN–BiLSTM trong benchmark được tuning lại dưới chính sách chung không
xử lý mất cân bằng; vì vậy giá trị `0.8380` không được trộn với kết quả nested CV
`0.8781` của mô hình final.

Kết luận đúng là các baseline ML cổ điển mạnh hơn CNN–BiLSTM trong benchmark
G1/G2 chuẩn hóa này. Dự án không tuyên bố CNN–BiLSTM vượt mọi baseline.

## Kiến trúc dữ liệu PostgreSQL-first

```text
CSV --(ingestion một lần)--> PostgreSQL dataset version
                              |-- source_records
                              |-- source_record_targets
                              |-- split ledger
                              |-- experiment runs
                              |-- predictions / metrics / recommendations
                                      |
                                      +--> final evidence bundle
```

CSV chỉ được đọc tại biên ingestion. Model selection, training, evaluation và
inference chính thức tải dữ liệu từ PostgreSQL theo `dataset_version_id`. Nhãn
được lưu riêng trong `source_record_targets`; pipeline dừng ngay nếu migration
003 hoặc target rows chưa đầy đủ.

## Cài đặt

Yêu cầu Python 3.10 và PostgreSQL. Cài dependency:

```powershell
py -3.10 -m pip install -r requirements-lock.txt
```

Tạo `.env` từ `.env.example` và điền thông tin kết nối. `DATABASE_URL` được ưu
tiên nếu có; nếu không, hệ thống dùng `POSTGRES_HOST`, `POSTGRES_PORT`,
`POSTGRES_DB`, `POSTGRES_USER` và `POSTGRES_PASSWORD`. File `.env` đã được
Git ignore và không được commit.

Áp dụng migrations theo thứ tự:

```text
database/migrations/001_create_source_ml_schema.sql
database/migrations/002_allow_append_only_recommendation_policy_versions.sql
database/migrations/003_add_source_record_targets.sql
```

Ingest dữ liệu một lần:

```powershell
py -3.10 scripts/ingest_dataset_to_postgres.py --dataset student-mat
```

## Chạy thực nghiệm

Chọn cấu hình CNN–BiLSTM bằng nested CV:

```powershell
py -3.10 scripts/optimize_model_selection.py --dataset student-mat `
  --dataset-version-id 1 --n-trials 30 --outer-folds 5 --inner-folds 3 `
  --selection-seed 42 --selection-run-id nested-full-20260710
```

Chạy mô hình final bằng cấu hình đã đóng băng:

```powershell
py -3.10 scripts/run_pipeline.py --dataset student-mat --target-mode 3class `
  --dataset-version-id 1 `
  --selection-config-json artifacts/model_selection/nested-full-20260710/selected_config.json
```

Chạy benchmark công bằng 7 mô hình:

```powershell
py -3.10 scripts/run_fair_model_comparison.py --dataset-version-id 1 `
  --run-id fair-model-comparison-full
```

Kết quả runtime của benchmark được tạo tại
`artifacts/baseline_comparison/<run-id>/` và được Git ignore để tránh commit các
file thực nghiệm lớn. Các con số chính đã được cố định trong README và
`PROJECT.md`.

Xác minh evidence cuối:

```powershell
py -3.10 scripts/verify_final_evidence.py
```

Chạy test:

```powershell
py -3.10 -m pytest -q
```

Trạng thái gần nhất: 87 passed, 5 skipped. Các test bị skip là test tích hợp phụ
thuộc thông tin kết nối PostgreSQL tại runtime.

## Hệ thống khuyến nghị

Khuyến nghị được tạo bởi policy luật xác định `student_mat_rule_policy_v3`,
không phải recommender học máy. Policy kết hợp dự đoán, độ tin cậy và ngữ cảnh
được phép để sinh yếu tố rủi ro, hành động ưu tiên, lý do và cảnh báo cần người
phụ trách xem xét. Đánh giá trên 79 đầu ra cho thấy schema hợp lệ, có giải thích,
có hành động cụ thể, không mâu thuẫn và không rò metadata nhạy cảm đều đạt 100%.
Đánh giá chuyên gia và hiệu quả can thiệp vẫn chưa được thu thập.

## Cấu trúc repository

```text
config/                 hợp đồng đặc trưng theo thời điểm dự báo
database/migrations/    schema và migrations PostgreSQL
src/                    pipeline dữ liệu, mô hình, đánh giá, khuyến nghị
scripts/                ingestion, selection, final run, benchmark, verification
tests/                  unit, protocol và PostgreSQL integration tests
artifacts/model_selection/  evidence lựa chọn cấu hình
artifacts/final/        evidence final đã đóng băng
docs/report_context/    dữ kiện và dàn ý hỗ trợ viết báo cáo
reports/scientific_audit/ kiểm toán leakage và nested CV
```

## Giới hạn sử dụng

- G2 là tín hiệu late-stage rất mạnh; hiệu năng cao không đồng nghĩa cảnh báo sớm.
- Chuỗi chỉ dài hai bước nên không được diễn giải là phụ thuộc thời gian dài hạn.
- Dữ liệu nhỏ, một môn học và một bối cảnh lịch sử; chưa đủ cho triển khai rộng.
- Điểm High hoàn hảo trên 15 mẫu test không phải bảo đảm tổng quát hóa.
- Dự đoán và khuyến nghị không phải kết luận nhân quả.
- Không dùng hệ thống để tự động xếp hạng, kỷ luật hoặc loại bỏ người học.

Xem [PROJECT.md](PROJECT.md) để có context học thuật đầy đủ phục vụ viết báo cáo.
