# Dự đoán kết quả học tập của sinh viên bằng CNN–BiLSTM

Repository này là mã nguồn và evidence của khóa luận về dự đoán kết quả học tập trên **ba bộ dữ liệu**: `student-mat`, `student-por` và OULAD. Dự án so sánh Machine Learning với Deep Learning, trong đó CNN–BiLSTM là kiến trúc chính cần kiểm chứng, không phải mô hình được mặc định phải thắng.

Tài liệu kỹ thuật đầy đủ và có thẩm quyền cao nhất của repository là [PROJECT.md](PROJECT.md).

## 1. Ba nghiên cứu trong đồ án

| Nghiên cứu | Bộ dữ liệu | Bài toán | Quy mô evidence chính | Kết luận chính |
| --- | --- | --- | ---: | --- |
| Study A | UCI `student-mat` | Dự đoán Low/Medium/High từ G1, G2 | 316 development records | Random Forest có điểm Macro-F1 cao nhất nhưng practical-tie với quy tắc G2 và SVM; quy tắc G2 được chọn làm overall model, CNN–BiLSTM được giữ làm thesis hybrid |
| Study B | UCI `student-por` | Lặp lại bài toán ba lớp với cohort và nested CV riêng | 649 records | Random Forest đạt Macro-F1 cao nhất; CNN–BiLSTM không vượt ML |
| Study C | OULAD | Nhận diện At-risk tại mốc giữa khóa từ chuỗi hoạt động theo tuần | 15.378 grouped-development records | CNN–BiLSTM Ensemble có point estimate cao nhất nhưng practical-tie với MLP |

`student-mat` có 395 dòng gốc, nhưng 79 dòng đã từng được quan sát trong quá trình phát triển nên được khóa dưới trạng thái `legacy_heldout_observed`; chúng không được dùng để tạo claim test-set chưa từng thấy.

## 2. Dữ liệu và target

### `student-mat` và `student-por`

- Input chính: G1 và G2.
- Target: G3 được chia thành Low (0–9), Medium (10–14), High (15–20).
- Metric chính: Macro-F1.
- G3, ID dòng, fold ID và metadata dự đoán không được dùng làm feature.

### OULAD

- Đơn vị quan sát: `(code_module, code_presentation, id_student)`.
- Target vận hành: At-risk = Withdrawn hoặc Fail; Not-at-risk = Pass hoặc Distinction.
- Mốc kết quả cuối: `F2_MIDDLE`, chỉ dùng sự kiện trong khoảng `0 <= date < cutoff`, với cutoff bằng 50% độ dài presentation.
- Split theo `global id_student`; cùng một sinh viên không xuất hiện ở cả train và validation.
- Demographic/sensitive attributes và sự kiện sau cutoff không đi vào mô hình chính.

## 3. Các mô hình được so sánh

Machine Learning:

- Logistic Regression
- Random Forest
- SVM
- HistGradientBoosting

Deep Learning:

- MLP
- CNN
- BiLSTM
- CNN–BiLSTM
- CNN–BiLSTM Ensemble

Mã kỹ thuật được giữ trong artifact và PostgreSQL để bảo toàn lineage; nội dung dành cho người đọc dùng [tên mô hình đơn giản](docs/THESIS_MODEL_TERMS.md).

## 4. Kiến trúc CNN–BiLSTM

Với UCI, CNN–BiLSTM nhận chuỗi điểm `[G1, G2]`. Chuỗi chỉ có hai timestep nên khả năng chứng minh ưu thế của CNN/BiLSTM bị giới hạn.

Với OULAD, mô hình nhận chuỗi hoạt động theo tuần. CNN trích xuất mẫu cục bộ, BiLSTM học quan hệ theo thứ tự, rồi biểu diễn temporal được kết hợp với thống kê tổng hợp và static context hợp lệ tại cutoff. Mô hình được huấn luyện bằng ba seed cố định; **CNN–BiLSTM Ensemble** là trung bình số học xác suất của cả ba lần huấn luyện, không phải một kiến trúc mạng mới.

## 5. Kết quả Study A — `student-mat`

| Model | Accuracy | Macro-F1 | High F1 | Macro PR-AUC | RMSE G3 | R² G3 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| G2 deterministic rule | 0.8924 | 0.8988 | 0.9246 | 0.8461 | 2.0086 | 0.8050 |
| Random Forest | 0.8924 | 0.9000 | 0.9332 | 0.9526 | 2.4609 | 0.7065 |
| SVM | 0.8829 | 0.8901 | 0.9246 | 0.9602 | 2.3605 | 0.7305 |
| CNN–BiLSTM | 0.8462 | 0.8504 | 0.8694 | 0.9510 | 2.4632 | 0.7067 |
| Ordinal CNN–BiLSTM | 0.8315 | 0.8383 | 0.8701 | 0.9457 | 2.4329 | 0.7128 |

Random Forest có point estimate Macro-F1 cao nhất. Tuy nhiên, Random Forest, quy tắc G2 và SVM nằm trong practical tie theo protocol. Quy tắc G2 được chọn làm **final overall model** theo tie-break và độ đơn giản; nominal CNN–BiLSTM là **final thesis hybrid model**. Hai vai trò này cố ý tách biệt.

## 6. Kết quả Study B — `student-por`

| Model | Accuracy | Macro-F1 | Macro PR-AUC |
| --- | ---: | ---: | ---: |
| Random Forest | 0.9014 | 0.8698 | 0.9315 |
| SVM | 0.8952 | 0.8659 | 0.9308 |
| HistGradientBoosting | 0.8968 | 0.8628 | 0.9329 |
| CNN–BiLSTM | 0.8752 | 0.8470 | 0.9273 |
| Logistic Regression | 0.8844 | 0.8449 | 0.9326 |
| G2 deterministic rule | 0.8428 | 0.8166 | — |

Random Forest là mô hình tốt nhất của independent nested evaluation trên `student-por`. Qua ba seed khai báo, Random Forest có Macro-F1 trung bình 0.8672 (SD 0.0023), còn CNN–BiLSTM có trung bình 0.8437 (SD 0.0151).

Cross-subject transfer từ `student-mat` sang `student-por` được báo riêng: CNN–BiLSTM đạt Macro-F1 0.8445, Random Forest 0.8250, SVM 0.8181 và quy tắc G2 0.8166. Đây là **frozen transfer/domain-shift analysis**, không phải external validation độc lập vì hai bộ UCI có các hồ sơ quasi-identity trùng nhau.

## 7. Kết quả Study C — OULAD

| Model | Macro-F1 | Risk Precision | Risk Recall | PR-AUC |
| --- | ---: | ---: | ---: | ---: |
| Logistic Regression | 0.8257 | 0.8286 | 0.7419 | 0.8875 |
| Machine Learning with Dynamic Features | 0.8260 | 0.8357 | 0.7349 | 0.8893 |
| MLP | 0.8287 | 0.8385 | 0.7390 | 0.8918 |
| CNN–BiLSTM | 0.8292 | 0.8195 | 0.7615 | 0.8923 |
| CNN–BiLSTM Ensemble | 0.8311 | 0.8406 | 0.7431 | 0.8927 |

CNN–BiLSTM Ensemble đạt point estimate Macro-F1 cao nhất. Chênh lệch so với MLP ensemble là 0.0025, thấp hơn superiority margin 0.005, nên verdict cuối là **PRACTICAL_TIE**. Nghiên cứu không tuyên bố Deep Learning vượt trội tuyệt đối.

Các biểu đồ trình bày dành cho khóa luận nằm trong [reports/thesis_figures](reports/thesis_figures).

## 8. Hệ thống khuyến nghị

Hệ thống khuyến nghị của Study A dùng CNN–BiLSTM năm seed làm nguồn model score và quy tắc G2 làm agreement guardrail. Pipeline tạo mục tiêu và kế hoạch bốn tuần theo luật chuyên gia, bắt buộc advisor review, có follow-up và revision bất biến.

- Technical validation: **PASS**.
- Expert validation: **PENDING**.
- Effectiveness validation: **NOT PERFORMED**.
- 316 cases phát triển; 71 cases (22,47%) đi vào uncertainty/agreement review.
- 0 action conflict, 0 duplicate action, 0 workload violation.

Không có claim rằng khuyến nghị làm tăng điểm hoặc có tác động nhân quả.

## 9. PostgreSQL và lưu vết khoa học

PostgreSQL lưu source identity, target tách biệt, cohort/split membership, prediction, metric và evidence registry. OULAD event/snapshot lớn được lưu bằng Parquet; PostgreSQL lưu metadata, checksum và lineage thay vì nhét toàn bộ event vào một JSON payload lớn.

Application role là least-privileged role, không phải superuser. Destructive integration test chỉ được chạy trên disposable database.

## 10. Cấu trúc repository

```text
project.py  lệnh duy nhất cho kiểm tra, figure, ingest và chuẩn bị dữ liệu
configs/    frozen scientific protocols và display-name mapping
database/   migrations, constraints và lineage schema
src/        model, feature, estimator, metric, PostgreSQL và recommendation code
scripts/    runner khoa học nội bộ, evidence registration và strict validators
tests/      unit, leakage, split, checksum, database và scientific-contract tests
artifacts/  immutable machine-readable scientific evidence
reports/    report mirrors, figures và scientific assessments
docs/       thuật ngữ và hướng dẫn vận hành cần thiết
```

Evidence trên GitHub được nhóm theo `student_mat`, `student_por`, `oulad` và `archive`, không trình bày theo mã phase nội bộ. Các runner Strategy A–E, locked-test materializer, wrapper chạy một lần và tài liệu kế hoạch V2/V3 đã được loại khỏi source tree cuối. Các runner nghiên cứu còn lại trong `scripts/` chỉ phục vụ tái lập evidence, không phải lệnh thường dùng. Immutable evidence cũ vẫn được giữ nguyên để audit; source trước khi đổi namespace có thể khôi phục từ tag `archive/pre-evidence-namespace-cleanup-20260716`.

## 11. Cách kiểm tra dự án

Cài môi trường:

```powershell
py -3.10 -m pip install -r requirements-lock.txt
Copy-Item .env.example .env
```

Xem nhanh ba evidence bundle chính, không train:

```powershell
py -3.10 project.py status
```

Kiểm tra toàn bộ evidence ba dataset, không train:

```powershell
py -3.10 project.py validate
```

Chạy full test suite:

```powershell
py -3.10 -m pytest -q
```

Tạo lại figure từ evidence đóng băng:

```powershell
py -3.10 project.py figures
```

Ingest UCI vào PostgreSQL:

```powershell
py -3.10 project.py ingest student-mat
py -3.10 project.py ingest student-por
```

Audit hoặc materialize OULAD mà không train:

```powershell
py -3.10 project.py audit-oulad
py -3.10 project.py prepare-oulad --resume
```

Không chạy trực tiếp các experiment file trong `scripts/` khi chỉ muốn kiểm tra repository; đó là expensive historical reproduction và phải dùng đúng frozen protocol.

## 12. Evidence chính

- `student-mat`: [final prediction evidence](artifacts/student_mat/final)
- `student-por`: [independent and transfer evidence](artifacts/student_por/final)
- OULAD: [final ensemble and PostgreSQL evidence](artifacts/oulad/final)
- Recommendation: [technical recommendation evidence](artifacts/student_mat/recommendation)
- Historical/smoke evidence: [archive index](artifacts/archive)

## 13. Hạn chế

- `student-mat` nhỏ và chỉ có hai grade timestep; 79 dòng đã quan sát không còn là locked test.
- `student-por` transfer có quasi-identity overlap với `student-mat`.
- OULAD final result là grouped development evidence tại mốc giữa khóa; future benchmark không được dùng trong closure cuối.
- OULAD primary closure là binary at-risk, không phải target bốn lớp.
- CNN–BiLSTM practical-tie với MLP trên OULAD; CNN incremental value chưa được chứng minh chắc chắn.
- Chưa có external unseen confirmation dataset.
- Expert review thật và prospective effectiveness study của recommendation chưa thực hiện.
