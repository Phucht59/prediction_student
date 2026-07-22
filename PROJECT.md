# Project Overview

## Research objective

Xây dựng hệ thống dự đoán kết quả học tập, nhận diện rủi ro và hỗ trợ khuyến nghị bằng bằng chứng có thể kiểm toán.

## Datasets

`student-mat` và `student-por` cung cấp bài toán Low/Medium/High. OULAD cung cấp bài toán Not-at-risk/At-risk theo khóa thời gian đã đăng ký.

## CNN-BiLSTM models

- CNN-BiLSTM — Student-Mat
- CNN-BiLSTM — Student-Por
- CNN-BiLSTM — OULAD

Ba tên trên là các model product riêng theo dataset, cùng API CNN-BiLSTM chính thức và checkpoint/evidence riêng.

## Machine Learning comparators

Mọi dataset dùng cùng thứ tự: CNN-BiLSTM, CNN-only, BiLSTM-only, Logistic Regression, Decision Tree, Random Forest, HistGradientBoosting, SVM và XGBoost. Thiếu frozen final evidence được ghi `N/A`.

## Prediction tasks

Hai dataset UCI là phân loại ba lớp; OULAD là phân loại rủi ro nhị phân và xếp hạng xác suất theo ngân sách Top-k.

## Recommendation system

Student Risk-Based Recommendation System sinh kế hoạch hỗ trợ từ risk profile, kiểm tra conflict, duplicate, workload và replay. Chỉ số cần nhãn chuyên gia vẫn chờ `PENDING_EXPERT_LABELS`.

## Data flow

Frozen outer-OOF/ensemble predictions → metric audit → risk profile → safeguarded recommendation → canonical JSON/CSV → reports.

## Evaluation protocol

Final metrics chỉ đến từ `FINAL_OUTER_OOF` hoặc `FINAL_PROBABILITY_ENSEMBLE`. Không dùng best seed, best fold hoặc inner screening làm kết quả cuối. Future OULAD luôn khóa.

## Final results

| Dataset | Official model | Macro-F1 |
|---|---|---:|
| student-mat | CNN-BiLSTM — Student-Mat | 0.9015 |
| student-por | CNN-BiLSTM — Student-Por | 0.8623 |
| oulad | CNN-BiLSTM — OULAD | 0.8281 |

Chi tiết và checksum ở `artifacts/final/` và `reports/final/`.

## Repository structure

Public configuration ở `configs/final`, API ở `src`, validation scripts ở `scripts/final`, canonical evidence ở `artifacts/final`, và báo cáo ở `reports/final`.

## Reproducibility

`python project.py final validate` dựng lại canonical payload từ frozen evidence, đối chiếu checksum, bảng, class metrics, future lock và expert status mà không train.

## Scientific limitations

Không tuyên bố hiệu quả can thiệp nhân quả, không ngoại suy sang miền chưa được kiểm định, và không điền metric thiếu bằng số ước lượng.
