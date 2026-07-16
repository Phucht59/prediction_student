# Hợp đồng kỹ thuật dự án

## Problem definition

Dự án dự đoán kết quả học tập từ thông tin có sẵn tại prediction cutoff. Hai bài toán chính là phân loại kết quả ba lớp trên UCI Student Performance và nhận diện sinh viên có nguy cơ trên chuỗi hoạt động OULAD. Recommendation là một contract riêng, chỉ tạo bản nháp non-causal cho advisor review.

## Scope

- UCI Mathematics: 395 bản ghi, trong đó 316 bản ghi development và 79 bản ghi legacy-observed không được dùng để xác nhận.
- UCI Portuguese: protocol, cohort, fold và evidence namespace riêng.
- OULAD: grouped development evaluation theo `global id_student`, snapshot tuần tại cutoff và target tách khỏi feature.
- Không có claim production-ready, external confirmation hoặc causal effectiveness.

## Data contracts

PostgreSQL là lineage system of record cho source identity, cohort, split, prediction, metric và evidence bundle. Dữ liệu sự kiện lớn có thể được materialize bằng Parquet; database lưu metadata, checksum và quan hệ truy vết.

Feature snapshot phải ghi cutoff, source hashes, feature-contract hash, channel order, sequence length và checksum. Target không được lưu trong feature snapshot.

## Target contracts

UCI Mathematics/Portuguese:

- Low: G3 từ 0–9.
- Medium: G3 từ 10–14.
- High: G3 từ 15–20.
- Macro-F1 là metric phân loại chính.

OULAD at-risk:

- At-risk: Withdrawn hoặc Fail.
- Not-at-risk: Pass hoặc Distinction.
- Nhãn, `date_unregistration` và sự kiện sau cutoff bị cấm làm feature.

## Feature contracts

- UCI late-stage primary input là G1/G2; G3 không bao giờ là input.
- OULAD temporal input chỉ dùng activity/assessment evidence đã có trước cutoff.
- Preprocessing, imputation, scaling, threshold selection và resampling chỉ fit trên training partition.
- Demographic/sensitive attributes không đi vào primary model; chúng chỉ được dùng cho fairness/error audit khi protocol cho phép.

## Model roles

| Display name | Role |
| --- | --- |
| Logistic Regression, Random Forest, SVM, HistGradientBoosting | Machine Learning baselines/comparators |
| MLP | Aggregate-only Deep Learning control |
| CNN | Temporal convolution ablation |
| BiLSTM | Recurrent-sequence ablation |
| CNN–BiLSTM | Thesis temporal hybrid architecture |
| CNN–BiLSTM Ensemble | Mean probability of all three declared CNN–BiLSTM seeds; proposed OULAD result |

Technical candidate IDs remain immutable keys in PostgreSQL and scientific evidence. User-facing output adds `display_name` without replacing those keys.

## Validation rules

1. Split theo student group; không random row split khi đánh giá unseen-student generalization.
2. Architecture, hyperparameter, epoch và threshold chỉ được chọn bên trong training/inner folds.
3. Không chọn seed tốt nhất; ensemble dùng đúng toàn bộ seed đã khai báo.
4. Probability phải finite, trong [0,1] và có tổng hợp lệ.
5. Macro-F1 là primary metric; Accuracy, Precision, Recall, F1 và PR-AUC là classification metrics phụ/guardrail.
6. RMSE/R² chỉ được dùng theo continuous-prediction contract, không mã hóa class thành 0/1/2 rồi gọi là G3 regression.
7. Không tạo composite score hậu nghiệm.
8. Evidence đã đóng băng không được sửa để khớp narrative mới.

## Official OULAD development results

| Model | Macro-F1 | Risk Precision | Risk Recall | PR-AUC |
| --- | ---: | ---: | ---: | ---: |
| Logistic Regression | 0.8257 | 0.8286 | 0.7419 | 0.8875 |
| Machine Learning with Dynamic Features | 0.8260 | 0.8357 | 0.7349 | 0.8893 |
| MLP | 0.8287 | 0.8385 | 0.7390 | 0.8918 |
| CNN–BiLSTM | 0.8292 | 0.8195 | 0.7615 | 0.8923 |
| CNN–BiLSTM Ensemble | 0.8311 | 0.8406 | 0.7431 | 0.8927 |

CNN–BiLSTM Ensemble có point estimate Macro-F1 cao nhất nhưng practical-tie với MLP. Không được chuyển kết quả này thành claim superiority.

## Evidence hierarchy

1. Official grouped development evidence và fair probability-ensemble closure.
2. PostgreSQL scientific closure với prediction/metric reproduction và least-privileged permission audit.
3. Independent UCI Portuguese evidence.
4. UCI Mathematics development freeze và recommendation technical evidence.
5. Historical, diagnostic, smoke, invalid-protocol và legacy-observed evidence chỉ dùng cho audit.

Artifact immutable phải giữ nguyên checksum, candidate registry, prediction và provenance.

## Recommendation governance

Pipeline: CNN–BiLSTM score + deterministic agreement reference → uncertainty/agreement assessment → feature governance → structured goals/actions → explanation → advisor decision → follow-up và immutable revision.

- Technical validation: PASS.
- Expert validation: PENDING.
- Effectiveness validation: NOT PERFORMED.
- Không recommendation nào tự động active.

## Prohibited claims

- CNN–BiLSTM đã được chứng minh vượt trội tuyệt đối so với Machine Learning/MLP.
- Mô hình đã được xác nhận trên external test hoàn toàn chưa thấy.
- Khả năng tổng quát hóa hoặc production readiness đã được chứng minh.
- Recommendation cải thiện điểm hoặc có causal effect.
- Expert validation đã pass khi chưa có rating thật.

## Repository conventions

- Internal candidate ID là lineage key; display name là presentation metadata.
- Artifacts dùng run ID và checksum; không overwrite official/historical evidence.
- Database credentials, dump, cache, runtime log và `.env` không được commit.
- Destructive migration test chỉ chạy trên disposable database bằng least-privileged app role.
- Quick validation không được tự động khởi chạy Optuna hoặc model training.

## Reproduction commands

```powershell
py -3.10 -m pip install -r requirements-lock.txt
py -3.10 -m pytest -q
py -3.10 scripts/validate_thesis_release.py
py -3.10 scripts/generate_thesis_figures.py
```

PostgreSQL integration tests cần `POSTGRES_TEST_DSN` và `POSTGRES_TEST_APP_DSN` trỏ tới disposable test setup. Không ghi credential vào source hoặc command examples.

## Known limitations

- Evidence hiện tại là development evidence, chưa có external unseen confirmation.
- Dữ liệu UCI nhỏ và chỉ có hai grade timestep.
- OULAD temporal result là practical tie với aggregate-only MLP.
- Context/sensitive features bị giới hạn bởi timing, semantic và fairness contracts.
- Expert review thật và prospective effectiveness study chưa thực hiện.

## Future research

1. External validation trên cohort/presentation hoàn toàn chưa quan sát.
2. Shadow/prospective study trước mọi claim effectiveness.
3. Expert review độc lập và adjudication cho recommendation casebook.
4. Nhiều mốc thời gian hơn để kiểm tra giá trị tăng thêm của temporal representation.
