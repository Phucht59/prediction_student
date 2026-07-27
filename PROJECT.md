# Project Overview

## Objective

Xây dựng mô hình học kết hợp để dự đoán thành tích học tập sinh viên, tạo hồ sơ
rủi ro và khuyến nghị có bằng chứng truy vết.

## Official models

- CNN-BiLSTM MAT (`cnn_bilstm_mat`)
- CNN-BiLSTM POR (`cnn_bilstm_por`)
- CNN-BiLSTM OULAD (`cnn_bilstm_oulad`)

Student-Mat và Student-Por là bài toán ba lớp Low/Medium/High. OULAD là bài toán
Not-at-risk/At-risk tại cutoff đã đăng ký.

## Data flow

Data → preprocessing → CNN-BiLSTM prediction → risk profile → safeguarded
recommendation → PostgreSQL/evidence.

Final metrics chỉ lấy từ complete outer-OOF probability ensembles. Không chọn
best seed/best fold, không dùng outer test để tuning và không truy cập Future
OULAD.

## Final results

| Dataset | Official model | Macro-F1 |
|---|---|---:|
| Student-Mat | CNN-BiLSTM MAT | 0.9015 |
| Student-Por | CNN-BiLSTM POR | 0.8623 |
| OULAD | CNN-BiLSTM OULAD | 0.8281 |

Canonical evidence nằm trong `artifacts/final`; báo cáo nằm trong
`reports/final`; cấu hình nằm trong `configs/final`.

## Claim boundary

Prediction và recommendation không phải tuyên bố causal effectiveness.
CNN-BiLSTM không được tuyên bố vượt trội phổ quát so với machine learning.
