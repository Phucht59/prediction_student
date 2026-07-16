# PROJECT.md — Hợp đồng kỹ thuật và khoa học cuối cùng

## 1. Vai trò của tài liệu

`PROJECT.md` là nguồn mô tả có thẩm quyền cao nhất cho trạng thái cuối của repository. README là bản giới thiệu ngắn; artifact và report bundle là evidence bất biến; protocol trong `configs/` là hợp đồng thực thi đã đóng băng. Khi có khác biệt:

1. Prediction/metric được lấy từ official immutable artifact.
2. Cohort, split, feature và target được lấy từ resolved protocol/manifest tương ứng.
3. `PROJECT.md` quyết định cách diễn giải khoa học hiện hành.
4. Historical report, smoke output, diagnostic output và observed evidence không được dùng để thay official result.

Tài liệu này mô tả **ba nghiên cứu dữ liệu riêng biệt** và một hệ thống khuyến nghị có quản trị. Nó không phải nhật ký các lần thử nghiệm.

## 2. Đề tài, mục tiêu và câu hỏi nghiên cứu

Tên đề tài trình bày: **Dự đoán kết quả học tập của sinh viên bằng CNN–BiLSTM**.

Mục tiêu kỹ thuật:

- Thu thập, kiểm tra, materialize và lưu vết dữ liệu học tập.
- So sánh Machine Learning và Deep Learning trên cùng information contract.
- Xây dựng CNN–BiLSTM cho dữ liệu điểm ngắn và dữ liệu hành vi theo tuần.
- Đánh giá bằng nested/grouped development evidence, không chọn seed/split có lợi.
- Tạo hệ thống khuyến nghị lộ trình học có rule policy, advisor review, follow-up và immutable revision.
- Đảm bảo prediction, metric, split, source và model bundle có thể truy vết qua checksum/PostgreSQL.

Câu hỏi nghiên cứu:

1. Trên UCI `student-mat`, mô hình nào dự đoán ba mức G3 tốt nhất khi chỉ dùng G1/G2?
2. Kết luận đó có lặp lại trên UCI `student-por` với cohort và tuning độc lập không?
3. Mô hình frozen từ `student-mat` thay đổi thế nào khi chuyển trực tiếp sang `student-por`?
4. Trên OULAD, chuỗi hành vi nhiều tuần có giúp CNN–BiLSTM tạo giá trị tăng thêm so với ML/MLP aggregate hay không?
5. Có thể biến prediction thành learning-path draft an toàn, giải thích được và có human review hay không?

Giả thuyết khoa học đúng là: **CNN–BiLSTM có tạo giá trị tăng thêm khi đầu vào là chuỗi hành vi học tập nhiều tuần so với các baseline dùng snapshot tổng hợp hay không?** Repository không giả định trước Deep Learning phải thắng.

## 3. Phạm vi và trạng thái cuối

| Study | Dataset | Primary task | Official status |
| --- | --- | --- | --- |
| A | UCI Student Performance — Mathematics (`student-mat`) | Three-class G3 classification từ G1/G2 | Development-selected and frozen |
| B | UCI Student Performance — Portuguese (`student-por`) | Independent three-class evaluation + frozen cross-subject transfer | PASS, transfer có overlap limitation |
| C | Open University Learning Analytics Dataset (OULAD) | Binary at-risk prediction tại `F2_MIDDLE` | Fair ensemble closure PASS; practical tie |
| Recommendation | Study A prediction evidence | Governed four-week learning-path draft | Technical PASS; expert PENDING; effectiveness NOT PERFORMED |

Ngoài phạm vi cuối:

- Không retrain để làm đẹp điểm.
- Không thay target, split, seed registry hoặc metric sau khi xem kết quả.
- Không dùng 79 `legacy_heldout_observed` như test set chưa từng thấy.
- Không dùng future OULAD benchmark để chọn architecture/threshold.
- Không có learned recommender, reinforcement learning hoặc causal recommender.
- Không có production-readiness, external-confirmation hoặc causal-effect claim.

## 4. Kiến trúc hệ thống tổng thể

```text
Raw sources
  ├─ UCI student-mat / student-por CSV
  └─ OULAD relational CSV tables
        ↓ source hashing + ingestion audit
PostgreSQL lineage registry + Parquet feature snapshots
        ↓ immutable cohort / split / feature / target contracts
Study-specific model evaluation
        ↓ OOF predictions + metrics + checksums
Scientific evidence registry
        ↓
Thesis tables / figures / governed recommendation policy
```

Nguyên tắc dữ liệu:

- PostgreSQL là system of record cho source identity, dataset version, split membership, prediction, metric và evidence registration.
- OULAD events/snapshots lớn dùng Parquet; PostgreSQL giữ locator, schema metadata, checksum và lineage.
- Target được lưu tách khỏi feature snapshot.
- Mỗi snapshot có cutoff, feature-contract hash, source hashes, channel order, sequence length và checksum.
- CSV chỉ được đọc tại ingestion/materialization boundary, không phải trong estimator core.

## 5. Study A — UCI `student-mat`

### 5.1 Dataset và cohort

- Source: `data/raw/student-mat.csv`.
- Tổng bản ghi gốc: 395.
- Development cohort chính thức: 316.
- Legacy-observed cohort: 79.
- 79 dòng đã từng bị quan sát trong lịch sử phát triển, vì vậy không còn đủ điều kiện làm untouched locked test.
- Official headline chỉ dùng nested development evidence trên 316 dòng.

### 5.2 Target contract

| Class | G3 interval |
| --- | --- |
| Low | 0–9 |
| Medium | 10–14 |
| High | 15–20 |

Primary input: `[G1, G2]`.

Không được dùng G3/G3_raw, target-derived feature, row index, source ID, fold ID, dataset version ID, prediction metadata hoặc recommendation metadata làm input. Context track đóng trong official comparison.

### 5.3 Protocol

- Five immutable outer folds.
- Three inner folds để chọn config.
- Macro-F1 là primary classification metric.
- Full-fold refit sau epoch selection.
- Canonical resolved configuration và một estimator factory cho inner/outer/final.
- Fixed/replayable learning-rate policy; không SWA; main neural dùng `drop_last=False`; không BatchNorm.
- Main-comparison seed registry cho neural candidates: 42, 123, 155.
- Final stability seed registry: 202601–202605; không chọn best seed.
- Calibration chỉ được fit từ inner-OOF; N0 calibration cuối bị reject.
- Practical-tie margin: absolute Macro-F1 difference < 0.01 hoặc paired uncertainty interval chứa 0.

### 5.4 Candidate roles

- Reference: deterministic G2 threshold rule.
- ML: Random Forest, SVM; Logistic/ordinal sanity models trong comparison lịch sử.
- Neural: nominal/ordinal CNN–BiLSTM, MLP controls, CNN-only và BiLSTM-only ablations.
- Overall ranking và thesis-hybrid ranking là hai quyết định độc lập.

### 5.5 Official final metrics

| Model | Role | Accuracy | Macro Precision | Macro Recall | Macro-F1 | Weighted-F1 | High F1 | Macro PR-AUC | RMSE G3 | R² G3 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| G2 deterministic rule | Final overall | 0.8924 | 0.9078 | 0.8935 | 0.8988 | 0.8925 | 0.9246 | 0.8461 | 2.0086 | 0.8050 |
| Random Forest | Practical-tie ML comparator | 0.8924 | 0.9079 | 0.8924 | 0.9000 | 0.8920 | 0.9332 | 0.9526 | 2.4609 | 0.7065 |
| SVM | Practical-tie ML comparator | 0.8829 | 0.9035 | 0.8798 | 0.8901 | 0.8829 | 0.9246 | 0.9602 | 2.3605 | 0.7305 |
| CNN–BiLSTM | Final thesis hybrid | 0.8462 | 0.8606 | 0.8535 | 0.8504 | 0.8450 | 0.8694 | 0.9510 | 2.4632 | 0.7067 |
| Ordinal CNN–BiLSTM | Research comparator | 0.8315 | 0.8435 | 0.8621 | 0.8383 | 0.8289 | 0.8701 | 0.9457 | 2.4329 | 0.7128 |

Interpretation:

- Random Forest có point estimate Macro-F1 cao nhất.
- Random Forest, G2 rule và SVM là practical tie theo protocol.
- G2 rule được chọn làm `final_overall_model` theo tie-break và simplicity, không phải do statistical superiority.
- Nominal CNN–BiLSTM là `final_thesis_hybrid_model`.
- N0 và N1 không có superiority rõ; ordinal learning chưa chứng minh cải thiện.
- BiLSTM-only practical-tie với CNN–BiLSTM; CNN chưa chứng minh incremental value trên chuỗi hai timestep.
- Residual, multitask, imbalance và context gates không được mở trong final Study A.

### 5.6 Continuous-G3 contract

Classification-only models tạo continuous estimate bằng class probabilities và class-conditional means chỉ fit trên training partition. G2 rule dùng `predicted_G3 = G2`. RMSE/R² là secondary analysis và không được trộn với Macro-F1 thành composite score.

## 6. Study B — UCI `student-por`

### 6.1 Dataset và protocol độc lập

- Source: `data/raw/student-por.csv`.
- Source hash được khóa trong `configs/extension_protocol_v1.yaml`.
- Cohort: 649 records.
- Input/target bins giống Study A: G1/G2 → Low/Medium/High từ G3.
- Five outer folds, three inner folds, folds/config riêng cho `student-por`.
- Primary metric: Macro-F1; practical margin 0.01.
- Default resampling: none; class weight chỉ được chọn trong inner training.
- Seed registry: 42, 2026, 3407; seed stability được chạy cho Random Forest và CNN–BiLSTM finalist.

Candidate registry gồm G2 rule, Logistic Regression, Random Forest, SVM, HistGradientBoosting, MLP, CNN, BiLSTM, nominal CNN–BiLSTM và ordinal CNN–BiLSTM.

### 6.2 Independent nested result

| Model | Accuracy | Balanced Accuracy | Macro-F1 | Weighted-F1 | Macro PR-AUC |
| --- | ---: | ---: | ---: | ---: | ---: |
| Random Forest | 0.9014 | 0.8554 | 0.8698 | 0.8994 | 0.9315 |
| SVM | 0.8952 | 0.8571 | 0.8659 | 0.8944 | 0.9308 |
| HistGradientBoosting | 0.8968 | 0.8462 | 0.8628 | 0.8943 | 0.9329 |
| CNN–BiLSTM | 0.8752 | 0.8464 | 0.8470 | 0.8752 | 0.9273 |
| Logistic Regression | 0.8844 | 0.8068 | 0.8449 | 0.8807 | 0.9326 |
| G2 deterministic rule | 0.8428 | 0.8314 | 0.8166 | 0.8473 | — |
| BiLSTM | 0.8459 | 0.7737 | 0.7958 | 0.8407 | 0.8754 |
| MLP | 0.6934 | 0.4183 | 0.4047 | 0.6019 | 0.7479 |
| Ordinal CNN–BiLSTM | 0.2943 | 0.5333 | 0.3608 | 0.1843 | 0.8098 |
| CNN | 0.6441 | 0.3333 | 0.2612 | 0.5046 | 0.6921 |

Random Forest là independent in-domain champion. Seed stability:

- Random Forest: mean Macro-F1 0.8672, SD 0.0023, min 0.8654.
- CNN–BiLSTM: mean Macro-F1 0.8437, SD 0.0151, min 0.8272.

Kết luận: trên một UCI subject thứ hai, compact CNN–BiLSTM vẫn không vượt ML khi sequence chỉ là G1/G2.

### 6.3 Frozen cross-subject transfer

Study A models được train/freeze từ 316 `student-mat` development records rồi inference trực tiếp trên 649 `student-por` rows; không tune bằng Portuguese labels.

| Frozen source model | Accuracy | Macro-F1 |
| --- | ---: | ---: |
| CNN–BiLSTM | 0.8737 | 0.8445 |
| Random Forest | 0.8567 | 0.8250 |
| SVM | 0.8444 | 0.8181 |
| G2 deterministic rule | 0.8428 | 0.8166 |

Overlap audit phân loại 358 conservative matched, 275 conservative unmatched và 16 ambiguous shared-key rows. Vì có quasi-identity overlap giữa UCI Mathematics/Portuguese, đây là domain-shift/transfer analysis, **không phải external validation hoàn toàn độc lập**.

## 7. Study C — OULAD

### 7.1 Raw data và observation unit

Raw release gồm bảy bảng: `courses`, `assessments`, `studentAssessment`, `studentInfo`, `studentRegistration`, `studentVle`, `vle`.

Observation grain: `(code_module, code_presentation, id_student)`.

Materialized horizons:

| Forecast | Fraction of presentation | Eligible rows |
| --- | ---: | ---: |
| F1_EARLY | 20% | 26.734 |
| F2_MIDDLE | 50% | 24.603 |
| F3_LATE | 80% | 23.034 |

Final fair closure tập trung vào `F2_MIDDLE`. Trong 24.603 eligible F2 rows:

- 15.378 historical-development rows (14.687 unique students) dùng cho grouped OOF.
- 1.061 rows bị loại vì student overlap với future candidate.
- 8.164 future-candidate rows không được dùng trong final closure.

Cutoff formula: `floor(module_presentation_length * forecast_fraction)`. Chỉ event thỏa `0 <= date < cutoff_day` được materialize. Đây là implementation thực tế; đề xuất H28 ban đầu không phải final contract.

### 7.2 Target

Primary final target là binary operational target:

- At-risk = Withdrawn hoặc Fail.
- Not-at-risk = Pass hoặc Distinction.

Original four-class values được giữ trong raw/target lineage nhưng four-class classification không phải final primary closure. Withdrawn không bị coi là một bậc ordinal thấp hơn Fail.

### 7.3 Feature contract

Base sequence có 16 channels:

- interaction: total clicks, active days, unique sites/activity types;
- content/forum/quiz/assessment-related clicks;
- submitted/late assessment counts và available-score count;
- cumulative mean/weighted score;
- days since activity, inactivity weeks và score-missing mask.

Dynamics-aware sequence thêm 31 causal channels (tổng 47, dưới guardrail 48): log magnitudes, week-over-week deltas, rolling two-week momentum, inactivity transitions, behaviour shares, score deltas và submission rates. Mọi feature ở tuần `t` chỉ dùng tuần hiện tại/quá khứ.

Vector controls dùng:

- 161 aggregate features từ base sequence;
- 279 summaries từ dynamic channels;
- tổng 440 numeric matched-vector features cộng static context hợp lệ.

Static context được phép: module, presentation season, previous attempts, studied credits, registration lead time và known presentation length. Demographic/sensitive features không đi vào primary model; `final_result`, `date_unregistration`, exact presentation identity và post-cutoff events bị cấm.

### 7.4 Split và selection

- Three outer `StratifiedGroupKFold` folds.
- Two inner grouped folds.
- Group key: global `id_student`.
- Seeds: 42, 2026, 3407; không chọn seed tốt nhất.
- Primary: pooled grouped-OOF binary Macro-F1.
- Operational guardrail: At-risk Recall tại Precision ≥ 0.75.
- Threshold fit trên pooled inner-OOF, freeze trước outer validation.
- Probability ensemble = arithmetic mean của đúng ba declared seeds.
- Paired bootstrap dùng `id_student`, không bootstrap từng row.
- Superiority margin cuối: 0.005 Macro-F1.
- Future benchmark: NOT EXECUTED trong final fair closure.

### 7.5 Model families

- ML: Logistic Regression; compact dynamic Logistic/HistGradientBoosting selection; V1 còn có Random Forest/SVM/HGB baselines.
- Aggregate neural: MLP.
- Temporal ablations: CNN-only, BiLSTM-only.
- Hybrid: CNN–BiLSTM sequence + aggregate + static.
- Final OULAD prediction candidate: three-seed CNN–BiLSTM probability ensemble.

### 7.6 Final fair ensemble metrics

| Model | Macro-F1 | Accuracy | Balanced Accuracy | Risk Precision | Risk Recall | Risk F1 | PR-AUC | Brier | NLL | ECE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Logistic Regression | 0.8257 | 0.8363 | 0.8203 | 0.8286 | 0.7419 | 0.7829 | 0.8875 | 0.1203 | 0.3807 | 0.0466 |
| Dynamic ML | 0.8260 | 0.8370 | 0.8197 | 0.8357 | 0.7349 | 0.7820 | 0.8893 | 0.1153 | 0.3659 | 0.0111 |
| MLP ensemble | 0.8287 | 0.8395 | 0.8225 | 0.8385 | 0.7390 | 0.7856 | 0.8918 | 0.1155 | 0.3678 | 0.0252 |
| CNN–BiLSTM | 0.8292 | 0.8384 | 0.8254 | 0.8195 | 0.7615 | 0.7895 | 0.8923 | 0.1150 | 0.3645 | 0.0214 |
| CNN–BiLSTM Ensemble | 0.8311 | 0.8417 | 0.8250 | 0.8406 | 0.7431 | 0.7888 | 0.8927 | 0.1145 | 0.3634 | 0.0195 |

Các giá trị Accuracy/Balanced Accuracy/Risk F1/Brier/NLL/ECE được đọc từ `ensemble_metrics.csv`; không dùng để thay Macro-F1 làm primary metric.

Fair paired interpretation:

- CNN–BiLSTM Ensemble − MLP ensemble = +0.002454 Macro-F1.
- CNN–BiLSTM Ensemble − Dynamic ML = +0.005165 Macro-F1.
- CNN–BiLSTM Ensemble − Logistic Regression = +0.005408 Macro-F1.
- So với strongest fair comparator (MLP), delta thấp hơn margin 0.005.
- Scientific verdict: `PRACTICAL_TIE`.
- Point estimate tốt hơn không đủ để tuyên bố absolute superiority.

## 8. Giải thích kiến trúc CNN–BiLSTM

### 8.1 UCI compact model

```text
[G1, G2]
→ compact Conv1D
→ activation / optional LayerNorm / dropout
→ compact BiLSTM
→ nominal hoặc ordered head
```

Giới hạn: sequence length chỉ bằng hai; kernel 2 có thể làm output temporal length còn một. Kết quả ablation xác nhận CNN incremental value chưa được thiết lập.

### 8.2 OULAD temporal model

```text
weekly causal sequence
→ temporal convolution
→ BiLSTM
→ mask-aware pooling
                      ┐
aggregate statistics ├→ compact fusion → binary logit
static context        ┘
```

Ensemble là trung bình probability của seed 42, 2026 và 3407. Ensemble không được gọi là kiến trúc mới và không chọn subset seed.

## 9. Metric contract chung

Classification metrics:

- Primary: Macro-F1 (three-class UCI hoặc binary OULAD theo study).
- Secondary/guardrails: Accuracy, Balanced Accuracy, per-class Precision/Recall/F1, weighted F1, confusion matrix, one-vs-rest PR-AUC.
- Probability metrics: Brier, NLL, ECE khi model có probabilistic output hợp lệ.
- Stability: fold, seed, module/presentation, worst-seed/worst-module và class collapse.

Regression metrics UCI:

- MAE, RMSE và R² chỉ dùng theo continuous-G3 mapping đã đăng ký.
- Không mã hóa Low/Medium/High thành 0/1/2 rồi gọi đó là G3 regression.
- R² âm được giữ nguyên; không gọi R² là accuracy.

Không tạo composite score hậu nghiệm từ classification, calibration và regression metrics.

## 10. Leakage, preprocessing và imbalance rules

1. Target/outcome không đi vào feature snapshot.
2. Imputation, scaling, feature selection, calibration, class weighting và resampling chỉ fit trên training partition.
3. Outer/future labels không chọn architecture, hyperparameter, epoch hoặc threshold.
4. Student grouping được dùng cho OULAD unseen-student generalization.
5. No best-seed selection.
6. SMOTE/ADASYN không được áp trực tiếp lên temporal sequence vì có thể tạo chuỗi phi thực tế.
7. Study A main comparison và Study B default dùng resampling `none`; imbalance branch chỉ được mở theo pre-registered gate.
8. OULAD sử dụng loss/class-weight policy trong inner search; sequence ordering không được “cứu” bằng comparator thiếu feature.

Đề cương có yêu cầu khảo sát SMOTE/ADASYN. Repository bảo toàn implementation/diagnostic context nhưng final model selection không dựa vào việc ép mở hai kỹ thuật này khi gate không đạt. Báo cáo phải trình bày đây là giới hạn/negative evidence, không được nói chúng đã cải thiện final model.

## 11. Recommendation system contract

Recommendation source là **Study A** frozen prediction bundle, không phải OULAD at-risk model.

```text
N0 five-seed CNN–BiLSTM model scores
+ R0 deterministic G2 agreement reference
→ prediction snapshot
→ uncertainty/agreement gate
→ feature governance
→ expert-guided rule policy
→ structured goals/actions
→ explanation + limitations
→ advisor approve/modify/reject/request-more-information
→ follow-up + immutable revision
```

R0 không có probability/confidence. N0 calibration bị reject; output phải gọi là model score/ensemble score, không phải absolute confidence.

Feature registry mặc định cho recommendation: G1, G2 và deterministic trajectory `G2-G1`. Context features không có verified timing/semantics bị disable. Missing/invalid value không được thay bằng 0/default risk.

Technical evaluation:

| Item | Result |
| --- | ---: |
| Development cases | 316 |
| Eligible normal draft gate | 245 |
| Uncertainty/agreement review cases | 71 |
| Gate review rate | 22.47% |
| Advisor approval required | 100% |
| Generated actions | 1,313 |
| Action conflicts | 0 |
| Duplicate actions | 0 |
| Workload violations | 0 |
| Goal/action/explanation completeness | 100% |
| Expert casebook | 60 cases / 23 strata |

Validation states:

- `technical_validation = PASS`
- `expert_validation = PENDING`
- `effectiveness_validation = NOT_PERFORMED`

Structural correctness không phải recommendation effectiveness. Không được bịa expert ratings hoặc nói hệ thống giúp tăng điểm.

## 12. PostgreSQL contract

Các migration phải chạy theo thứ tự. Nhóm schema chính:

- source dataset versions, source records và target lineage;
- experiment runs, splits, predictions, metrics và recommendation rows;
- recommendation policy/feature/action/revision/advisor/follow-up tables;
- OULAD snapshot/evidence registry;
- fair ensemble evidence metadata và lineage-integrity triggers.

Yêu cầu:

- Stable source identity; foreign keys và uniqueness constraints đầy đủ.
- Completed run/evidence immutable hoặc append-only.
- Migration trong advisory-locked transaction; dry-run rollback trước commit.
- Backup trước database write có tác động.
- App role `NOSUPERUSER`, `NOCREATEDB`, `NOCREATEROLE`; không dùng `postgres` làm app-role evidence.
- Credentials không xuất hiện trong source, artifact hoặc log.
- Production database không dùng cho destructive integration testing.
- Fair closure database reproduction: 123.024 prediction rows, max probability delta 0, metric delta tối đa khoảng machine precision.

Hướng dẫn migration/recovery: `docs/MANUAL_POSTGRESQL_MIGRATION.md`.

## 13. Evidence hierarchy và immutable paths

### Official Study A

- Final development metrics/roles: `artifacts/final_repository_closure/final-repository-closure-corrected-20260715-6ab785d/`.
- Main comparison: `artifacts/strategy_b_phase_c/strategy-b-phase-c-20260714-5d34a66/`.
- Final stability/freeze: `artifacts/strategy_b_phase_e_prediction/strategy-b-phase-e-prediction-20260714-9007144/`.
- Recommendation technical validation: `artifacts/strategy_b_phase_d_recommendation/strategy-b-phase-d-recommendation-20260715-407ac0f/`.

### Official Study B

- `artifacts/study_b_student_por/study-b-student-por-20260715-v1/`.
- Mirrored report: `reports/study_b_student_por/study-b-student-por-20260715-v1/`.

### Official Study C

- Baseline multi-horizon evidence: `artifacts/study_c_oulad/study-c-oulad-20260715-v1/`.
- Exploratory temporal evidence: `artifacts/study_c_oulad_v3/oulad-deep-v3-f2-20260716-v1/`.
- Final fair ensemble/PostgreSQL closure: `artifacts/study_c_oulad_v3_closure/oulad-v3-fair-db-closure-20260716-v1/`.

### Historical/non-headline categories

- 79 observed rows and old `artifacts/final/*` outputs.
- Invalid fair-DL rows thiếu resolved fixed constants.
- Old estimator results trước corrected full-partition refit.
- Smoke runs, residual diagnostics và incomplete/recovery runs.

Historical evidence không bị xóa. Code runner một lần có thể bị loại khỏi `main` sau khi đã có immutable bundle và recovery tag; evidence vẫn giữ nguyên checksum.

## 14. Source tree cuối

Active high-level modules:

- `project.py`: entrypoint duy nhất cho các thao tác thường dùng; không cung cấp lệnh training.
- `src/models/student_grade.py`: compact UCI MLP/CNN/BiLSTM/CNN–BiLSTM/ordinal candidates.
- `src/studies/student_por/`: `student-por` data, model and evaluation contracts.
- `src/studies/oulad/`: base OULAD materialization/model contracts.
- `src/studies/oulad_v2/`, `oulad_v3/`: frozen temporal/aggregate implementations required to reproduce final OULAD evidence.
- `src/studies/oulad_v3_closure/`: fair probability-ensemble and closure logic.
- `src/model_selection.py`, `estimator_factory.py`, `train_pipeline.py`: reusable estimator primitives.
- `src/governed_recommendation.py`: canonical recommendation builder.
- `src/postgres_data_source.py`, `src/evaluation/`: PostgreSQL and evidence persistence.

Thao tác thường dùng đi qua `project.py`. Các file còn lại trong `scripts/` là runner khoa học nội bộ, evidence registration hoặc strict validator. Old Strategy A–E orchestration, extension wrapper chạy một lần, locked-test pipeline, final-closure generator, one-time cleanup script và plan markdown đều vắng mặt khỏi `main`.

Frozen configs named V2/V3 remain because they are **scientific protocol artifacts referenced by checksum**, not unfinished planning documents. Removing or rewriting them would break lineage.

## 15. Reproduction and validation commands

### 15.1 Quick validation — no training

```powershell
py -3.10 project.py status
py -3.10 project.py validate
py -3.10 project.py figures
git diff --check
```

`project.py validate` gọi strict release validator để kiểm tra checksum và headline metrics của cả ba dataset, sau đó chạy OULAD closure validator ở chế độ check-only. Không command nào trong quick validation train model.

### 15.2 Full test suite

```powershell
py -3.10 -m pytest -q
```

PostgreSQL destructive tests require isolated disposable admin/app DSNs. Skipped tests must remain skip, not fake pass.

### 15.3 Ingestion

```powershell
py -3.10 project.py ingest student-mat
py -3.10 project.py ingest student-por
```

OULAD audit/materialization:

```powershell
py -3.10 project.py audit-oulad
py -3.10 project.py prepare-oulad --resume
```

### 15.4 Expensive historical reproduction

Các lệnh sau có thể chạy nested CV/Optuna và không phải quick validation:

- `scripts/student_por_experiment.py`: independent nested evaluation của Study B.
- `scripts/oulad_experiment.py`: baseline multi-horizon của Study C.
- `scripts/oulad_tuning.py`: tuned aggregate/temporal comparator stage.
- `scripts/oulad_temporal.py`: temporal dynamics exploratory stage.
- `scripts/oulad_final_ensemble.py`: fair three-seed ensemble closure.

Các file hỗ trợ còn lại được đặt tên theo đúng chức năng:

- `scripts/database_audit.py`, `database_register_evidence.py`: audit schema và đăng ký evidence PostgreSQL.
- `scripts/validate_oulad_tuning.py`, `validate_oulad_temporal.py`, `validate_oulad_final.py`: validator cho từng tầng evidence OULAD.
- `scripts/validate_release.py`: release validator cuối cho cả `student-mat`, `student-por` và OULAD.

Các wrapper cũ cho ingest, OULAD audit, materialization, split, figure và validation đã được gộp vào `project.py`; chúng không còn tồn tại thành nhiều file rời.

Chỉ chạy với source data, environment, protocol, manifest và compute budget đã khóa. Không mở lại model selection trong final repository closure.

## 16. Test contract

Test suite bao phủ:

- target/feature separation và no leakage;
- inner/outer group disjointness;
- causal cutoff/dynamic features;
- probability finite/range/sum;
- exact three-seed ensemble;
- checkpoint replay và OOF metric recomputation;
- threshold inner-only;
- immutable evidence checksums;
- display-name mapping giữ nguyên candidate ID;
- recommendation missing/stale/conflict/workload/revision safety;
- PostgreSQL schema, target lineage, evidence registration, least-privileged permissions và reproduction;
- final README/PROJECT three-dataset truth.

Test count không được hard-code trong tài liệu. Báo cáo release phải lấy passed/skipped/failed từ output thật tại commit cuối.

## 17. Security, privacy và ethics

- Không commit `.env`, password, DSN, database dump hoặc raw secret.
- Không dùng sensitive demographics trong primary prediction/recommendation.
- Student/source IDs dùng cho grouping/lineage, không làm predictive feature.
- Model output hỗ trợ quyết định, không thay advisor.
- Không diễn giải association là cause.
- Không tự động kích hoạt recommendation cho sinh viên.
- Error/fairness audit phải báo subgroup limitations nhưng không tối ưu hậu nghiệm để che subgroup yếu.

## 18. Claims được phép

- Ba dataset đã được triển khai với protocol/evidence namespace riêng.
- Random Forest/G2 rule mạnh hơn compact CNN–BiLSTM trên hai UCI G1/G2 studies.
- OULAD tạo chuỗi nhiều tuần phù hợp hơn để kiểm tra temporal representation.
- CNN–BiLSTM Ensemble có point-estimate Macro-F1 cao nhất trong final OULAD fair comparison.
- OULAD CNN–BiLSTM Ensemble practical-tie với MLP; absolute superiority chưa được chứng minh.
- Recommendation policy vượt qua technical safety validation và luôn yêu cầu advisor review.

## 19. Claims bị cấm

- “CNN–BiLSTM chắc chắn vượt Machine Learning.”
- “Deep Learning cho kết quả tốt nhất trên cả ba dataset.”
- “79 dòng UCI là untouched test.”
- “`student-por` transfer là external validation độc lập.”
- “OULAD future benchmark là unseen final test.”
- “CNN có causal/incremental value đã được chứng minh chắc chắn.”
- “Khuyến nghị làm tăng điểm hoặc đã scientifically proven effective.”
- “Expert validation đã PASS.”
- “Model production-ready hoặc generalization proven.”

## 20. Known limitations

1. `student-mat` development cohort nhỏ; chỉ hai timestep và G2 gần G3.
2. 79 rows bị quan sát, nên không còn untouched confirmation.
3. `student-mat` và `student-por` có quasi-identity overlap.
4. `student-por` neural candidates có seed variance lớn hơn RF.
5. OULAD final fair evidence chỉ dùng historical-development F2; future candidate không được dùng để xác nhận.
6. OULAD primary final task là binary at-risk, không phải original four-class task.
7. Dynamic/sequence gain so với MLP chưa vượt superiority margin.
8. Context/sensitive features bị giới hạn bởi timing, semantics và fairness.
9. Không có external unseen dataset hoàn toàn độc lập.
10. Recommendation expert review và prospective effectiveness study chưa thực hiện.

## 21. Future research

Ưu tiên hợp lệ sau khóa luận:

1. Thu thập cohort/presentation mới hoàn toàn chưa quan sát để external confirmation.
2. Chạy prospective/shadow evaluation trước mọi deployment claim.
3. Thu thập expert ratings thật cho 60-case casebook và adjudication.
4. Đánh giá four-class OULAD như study riêng, không thay target sau khi xem binary result.
5. Kiểm tra temporal incremental value trên nhiều horizon bằng protocol đăng ký trước.
6. Chỉ khảo sát SMOTE/ADASYN trên vector representation và chỉ trong inner training; không tạo synthetic temporal sequence tùy tiện.

## 22. Git và preservation policy

- Final integration branch: `main`.
- Các branch thí nghiệm được xóa sau khi nội dung cần thiết đã merge và recovery tag đã push.
- Recovery tag trước source prune: `archive/pre-final-source-prune-20260716`.
- Không rewrite Git history, không force-push.
- Immutable scientific evidence không bị chỉnh sửa hoặc xóa.
- DOCX khóa luận không thuộc nhiệm vụ source closure này.

## 23. Final scientific truth

- **Study A overall model:** G2 deterministic rule, được chọn trong practical tie bằng simplicity tie-break.
- **Study A thesis hybrid:** nominal CNN–BiLSTM five-seed development-frozen model.
- **Study B champion:** Random Forest; CNN–BiLSTM không vượt ML.
- **Study C point-estimate leader:** CNN–BiLSTM Ensemble; verdict practical tie với MLP.
- **Recommendation:** governed, rule-based, non-causal, advisor-in-the-loop; technical PASS, expert PENDING, effectiveness NOT PERFORMED.
- **Validation scope:** development evidence; chưa có untouched external confirmation.
