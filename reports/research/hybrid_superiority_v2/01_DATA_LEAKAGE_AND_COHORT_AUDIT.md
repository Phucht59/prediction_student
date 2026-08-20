# 01 — Data leakage and cohort audit

Protocol v2 xây feature từ `data/raw` trong repo. Không đọc `C:\hufit\kltn`.

## UCI

| Kiểm tra | Thiết kế | Test |
|---|---|---|
| G3 không predictor | forbidden list + context numeric không có G3 | `test_g3_never_predictor` |
| S0 không G1/G2 | temporal mask all False | `test_uci_stage_availability_and_no_grade_in_aggregate` |
| S1 chỉ G1 | mask[:,0] True, mask[:,1] False | cùng test |
| S2 thứ tự G1 rồi G2 | temporal[:,0]=G1/20, [:,1]=G2/20 | `test_uci_s2_order_is_g1_then_g2` |
| G1/G2 không vào Hybrid aggregate | `aggregate_available=0`, aggregate zeros | cùng test |
| Absences | không có trong context Hybrid/baseline Panel A | protocol |
| Group split | StratifiedGroupKFold trên `global_student_group`; MAT/POR cùng proxy không tách fold | `make_splits` |
| Quasi-identity | không phải student ID thật; collision có thể gộp người khác | limitation, không che |

Baseline Panel A **được** G1 (S1) và G1+G2 (S2) như cột tabular. Đây là cùng thông tin thô, khác representation.

## OULAD

| Kiểm tra | Thiết kế |
|---|---|
| Cấm final_result/score/date_unregistration | không đưa vào feature frame |
| Event `t < cutoff` | `filter_events_cutoff_safe` |
| Unregistration | chỉ eligibility (risk-set), không phải predictor |
| 100% | full-information benchmark, không early warning |
| Cohort | primary = operational risk-set từng cutoff (N/prevalence khác nhau). Common-cohort là sensitivity |
| Length≈Withdrawn | `diagnose.length_shortcut_oulad`; không dùng shortcut để claim temporal causality |

`final_result` có thể nằm trong context parquet để audit Fail/Withdrawn, **không** vào `baseline_frame` predictors.

## Cohort OULAD đã đo (risk-set)

Static join: 32593 enrollments / 28785 students / prevalence 0.528.

| Cutoff | N eligible | Prevalence | Withdrawn still in set | max T (weeks) |
|---|---:|---:|---:|---:|
| 20% | 26697 | 0.424 | 4277 | 8 |
| 35% | 25606 | 0.399 | 3180 | 14 |
| 50% | 24599 | 0.375 | 2171 | 20 |
| 75% | 23159 | 0.336 | 731 | 29 |
| 100% | 22522 | 0.317 | 94 | 39 |

Không được vẽ 5 số AP như cùng một cohort. 100% operational gần như Fail-vs-success vì người Withdrawn đã rời trước cutoff. Shortcut length→Withdrawn là **sensitivity trên toàn bộ enrollment**, không phải panel 100% chính.

Split hashes: xem `artifacts/research/hybrid_superiority_v2/manifests/data_lock.json`.

## Split firewall

- Outer 3-fold, seed 42.
- Phát triển: loại test của outer fold 0.
- FIT/STOP/VALID rời nhóm.
- Preprocessor/resampler/teacher fit trên FIT (teacher OOF trong FIT).
- Threshold/calibration trên STOP.
- VALID cho Optuna.
- Outer test không feedback.

## SMOTE/ADASYN

Không nội suy one-hot+aggregate+sequence. SMOTENC chỉ có thể thử cho baseline tabular trong FIT. Hybrid: pos_weight / ranking / (focal) ladder.
