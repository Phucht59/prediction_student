# Dự đoán kết quả học tập của sinh viên bằng CNN–BiLSTM

## 1. Tên đề tài

Repository xây dựng và đánh giá các mô hình dự đoán kết quả học tập, trong đó CNN–BiLSTM là kiến trúc Deep Learning chính của khóa luận. Hệ thống cũng có pipeline khuyến nghị lộ trình học dựa trên luật chuyên gia, luôn yêu cầu giảng viên/cố vấn duyệt.

## 2. Mục tiêu

- Chuẩn hóa dữ liệu học tập và lưu vết nguồn dữ liệu trong PostgreSQL.
- So sánh công bằng Machine Learning và Deep Learning trên cùng cohort, feature contract và split.
- Kiểm tra CNN–BiLSTM có khai thác được chuỗi hành vi học theo tuần hay không.
- Tạo khuyến nghị có cấu trúc, giải thích, kiểm soát tải học và theo dõi phiên bản.
- Bảo toàn khả năng tái lập bằng manifest, checksum, prediction và metric đã đăng ký.

## 3. Bộ dữ liệu

Repository chứa ba phạm vi nghiên cứu:

- **UCI Student Performance – Mathematics:** bài toán ba lớp Low/Medium/High từ điểm G3, dùng G1 và G2 làm đầu vào giai đoạn muộn. Trong 395 bản ghi, 316 bản ghi thuộc development protocol; 79 bản ghi `legacy_heldout_observed` đã từng được quan sát nên không được dùng như test set chưa thấy.
- **UCI Student Performance – Portuguese:** cohort và lineage riêng, dùng để kiểm tra khả năng lặp lại protocol trên môn học khác.
- **OULAD:** dữ liệu tương tác học trực tuyến theo tuần. Kết quả chính bên dưới là bài toán nhận diện sinh viên có nguy cơ tại mốc giữa khóa, đánh giá bằng grouped development out-of-fold evidence.

Target, feature snapshot và split membership được tách rời. Dữ liệu sau cutoff, nhãn tương lai và các thuộc tính nhạy cảm không đi vào mô hình chính.

## 4. Các mô hình được so sánh

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

Các mã kỹ thuật vẫn được giữ trong database, artifact và prediction registry để truy vết. Tài liệu và biểu đồ dùng [tên mô hình dễ đọc](docs/THESIS_MODEL_TERMS.md).

## 5. Kiến trúc CNN–BiLSTM

CNN trích xuất các mẫu cục bộ trong chuỗi hoạt động theo tuần. BiLSTM học quan hệ theo thứ tự trong phần chuỗi đã quan sát. Biểu diễn chuỗi được kết hợp với các thống kê tổng hợp hợp lệ tại cutoff và static context không nhạy cảm.

CNN–BiLSTM được huấn luyện ba lần với ba seed cố định. **CNN–BiLSTM Ensemble** là trung bình số học xác suất dự đoán của cả ba lần huấn luyện; ensemble không phải một kiến trúc mạng mới.

## 6. Kết quả chính

Số liệu được đọc từ fair closure evidence, không tính lại hay chép từ một lần huấn luyện mới.

| Model | Macro-F1 | Risk Precision | Risk Recall | PR-AUC |
| --- | ---: | ---: | ---: | ---: |
| Logistic Regression | 0.8257 | 0.8286 | 0.7419 | 0.8875 |
| Machine Learning with Dynamic Features | 0.8260 | 0.8357 | 0.7349 | 0.8893 |
| MLP | 0.8287 | 0.8385 | 0.7390 | 0.8918 |
| CNN–BiLSTM | 0.8292 | 0.8195 | 0.7615 | 0.8923 |
| CNN–BiLSTM Ensemble | 0.8311 | 0.8406 | 0.7431 | 0.8927 |

CNN–BiLSTM Ensemble đạt point estimate Macro-F1 cao nhất. Tuy nhiên, chênh lệch so với MLP ensemble nằm trong vùng practical tie. Vì vậy, nghiên cứu **không tuyên bố CNN–BiLSTM vượt trội tuyệt đối**.

Ba biểu đồ dành cho khóa luận nằm trong [`reports/thesis_figures`](reports/thesis_figures). Comparator dùng feature động không được gán tên một thuật toán duy nhất vì các outer fold đã chọn hai họ estimator khác nhau.

## 7. Cách chạy dự án

Cài môi trường:

```powershell
py -3.10 -m pip install -r requirements-lock.txt
Copy-Item .env.example .env
```

Chạy toàn bộ test:

```powershell
py -3.10 -m pytest -q
```

Kiểm tra evidence mà không train lại mô hình:

```powershell
py -3.10 scripts/validate_thesis_release.py
```

Tạo lại figure từ evidence đã đóng băng:

```powershell
py -3.10 scripts/generate_thesis_figures.py
```

Ingest UCI vào một database đã được cấp quyền:

```powershell
py -3.10 scripts/ingest_dataset_to_postgres.py --dataset student-mat
```

Không dùng database production cho destructive integration tests. Các runner nested-CV/Optuna là lịch sử thí nghiệm tốn chi phí, không phải lệnh kiểm tra nhanh.

## 8. Cấu trúc thư mục

```text
configs/    protocol và mapping tên hiển thị
database/   migrations và ràng buộc PostgreSQL
src/        model, feature, metric, lineage và recommendation policy
scripts/    ingestion, validator, evidence utilities và historical runners
tests/      unit, contract và PostgreSQL integration tests
artifacts/  scientific evidence bất biến
reports/    báo cáo, figure và thesis context
docs/       tài liệu kỹ thuật và thuật ngữ
```

## 9. PostgreSQL

PostgreSQL lưu source identity, cohort/split membership, feature snapshot, prediction, metric và evidence bundle. `candidate_id` kỹ thuật không bị đổi; `display_name` chỉ là lớp trình bày.

Migration phải được chạy theo thứ tự và bằng role có quyền phù hợp. App role phải là least-privileged role, không dùng superuser để làm permission test xanh giả.

## 10. Kiểm thử

Test suite bao gồm:

- split/group và chống leakage;
- probability, checkpoint và metric recomputation;
- tính bất biến của scientific evidence;
- model display-name mapping;
- PostgreSQL schema, lineage, permission và reproduction;
- recommendation safety, revision và advisor-review lifecycle.

PostgreSQL destructive tests chỉ chạy khi có disposable DSN. Test bị skip phải được báo là skip, không chuyển thành pass giả.

## 11. Hạn chế

- Kết quả OULAD là grouped development evidence; benchmark presentation tương lai đã quan sát không phải external test chưa thấy.
- Chênh lệch CNN–BiLSTM Ensemble và MLP là practical tie, không phải superiority đã xác nhận.
- UCI Mathematics chỉ có hai mốc G1/G2, nên giá trị tăng thêm của CNN trên chuỗi ngắn chưa được chứng minh.
- 79 bản ghi đã quan sát không được dùng cho claim locked-test.
- Recommendation là rule-based, non-causal; expert validation còn pending và effectiveness chưa được đánh giá.
- Chưa có bộ dữ liệu external hoàn toàn chưa thấy để xác nhận khả năng tổng quát hóa.
