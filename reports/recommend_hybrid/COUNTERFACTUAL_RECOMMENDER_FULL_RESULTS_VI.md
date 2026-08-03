# FULL-COHORT COUNTERFACTUAL VALIDATION

## Trạng thái phát hành

- Branch: `codex/constrained-counterfactual-recommender`
- PR: `#4` — Draft, chưa merge
- Commit authority của lần đánh giá: `851e169340e23fd9af75e8f6ee0fd4232e079e7d`
- Claim boundary: `MODEL_ESTIMATED_RISK_REDUCTION_NOT_CAUSAL_EFFECT`
- Candidate release status: `CANDIDATE_VALIDATED_PENDING_EXPERT_REVIEW`
- Merge allowed: **NO**
- FINAL_RELEASE: **NO**

## Input authority và tính tái lập

Full cohort dùng đúng residual checkpoint release manifest, code tree, raw manifest, split manifest, recommendation manifest, action policy và preprocessor fingerprints được khóa trong `full_evaluation_input_registry.json`. Không retrain, không đổi model, checkpoint, split, preprocessing hay action definitions.

Lần chạy gồm 3 outer folds, 4 stages và ensemble 5 seed (`42, 1201, 2026, 3407, 7319`), tổng cộng 12 batch atomic. Runner có progress registry và resume-safe; tất cả batch đã hoàn tất, không có batch thiếu hoặc partial final claim.

## Kết quả full cohort

| Chỉ số | Kết quả |
|---|---:|
| Full-cohort records | 62.525 |
| Scored records | 41.472 |
| Scored coverage | 66,3287% |
| Fallback records | 21.053 |
| Fallback rate | 33,6713% |
| Mean top-action risk reduction | 0,111392 |
| Median top-action risk reduction | 0,073885 |
| Success@0,01 | 100,000% (41.472/41.472 scored) |
| Success@0,02 | 87,6712% (36.359/41.472 scored) |
| Success@0,03 | 78,5108% (32.560/41.472 scored) |
| Success@0,05 | 63,6309% (26.389/41.472 scored) |
| Success@0,10 | 39,0360% (16.189/41.472 scored) |
| Threshold crossing | 20,0589% (3.679/18.341 denominator) |
| Top-1 action concentration | 58,0440% |
| Action diversity | 5 |

`Mean top-action risk reduction` là chênh lệch rủi ro do model ước lượng giữa baseline và counterfactual action tốt nhất trên candidate set. Đây không phải xác suất thành công thực tế, treatment effect hay causal effect.

Fallback breakdown: `NO_ACTION_MET_MINIMUM_RISK_REDUCTION` = 12.750; `POLICY_ABSTAINED` = 8.303. Full evaluator chưa persist đủ cờ assessment availability độc lập; audit ghi nhận mục này là `NOT_PERSISTED`, không suy diễn thêm từ outcome.

## Audit và độ ổn định

- Success metric audit: **PASS**. Không có duplicate identity, leakage field hoặc scored row dưới ngưỡng 0,01; fallback bị loại khỏi success denominator một cách minh bạch; top action join coverage = 100%.
- Coverage/fallback audit: **PASS**, theo fold, stage, course/presentation và baseline-risk decile. Coverage thấp nhất ở decile rủi ro thấp D1 (28,5943%) và tăng rõ ở các decile rủi ro cao hơn.
- Fold stability: coverage lần lượt 67,1748% / 65,1442% / 66,6651%; mean reduction 0,110948 / 0,111150 / 0,112073.
- Stage stability: coverage EARLY_20 = 57,1955%; EARLY_35 = 62,6552%; MIDDLE_50 = 75,3999%; LATE_75 = 71,1377%. Mean reduction tương ứng 0,103537 / 0,115526 / 0,108541 / 0,117751.
- Seed stability: `DESCRIPTIVE_NOT_PER_SEED`. Evaluator dùng ensemble 5 seed đã đăng ký; không chạy single-seed thay thế vì sẽ làm thay đổi authority. Đây là limitation cần giữ trong luận văn.
- Deterministic replay: **PASS** trên 12 batch atomic, thứ tự fold/stage/seed cố định, identity duplicate-free và checksum registry.
- Protected-feature violations: **0 phát hiện trong artifact audit**.
- Leakage violations: **0 phát hiện trong artifact audit**.

## Baseline comparison

Tất cả baseline dùng cùng eligible ranked candidate set và cùng 41.472 dòng scored:

| Strategy | Mean reduction | Success@0,05 | Action concentration |
|---|---:|---:|---:|
| Existing policy ordering | 0,111392 | 63,6309% | 58,0440% |
| Fixed seeded random ordering | 0,115358 | 63,3391% | 52,8718% |
| Workload-only ordering | 0,106406 | 60,8579% | 71,0214% |
| Counterfactual risk ordering | 0,140160 | 68,9140% | 83,3116% |

Đây là so sánh descriptive trên model-estimated risk ordering, không phải bằng chứng causal. Risk ordering có concentration cao hơn, do đó cần expert review về tính hợp lý và workload trước mọi release.

## Expert review

Đã chuẩn bị **160 case** (98 scored, 62 fallback), có stratification theo stage, scored/fallback, fallback reason và baseline-risk band. Case chỉ chứa tín hiệu quan sát trước cutoff, candidate actions, reason codes, workload, evidence/uncertainty và model-estimated risks; không chứa future outcomes, protected attributes hoặc student identifiers.

Expert review chưa hoàn tất. Mỗi case cần ít nhất hai expert review độc lập; bất đồng chuyển adjudicator. Candidate chỉ có thể chuyển trạng thái sau khi review hoàn tất và các gate liên quan đạt yêu cầu.

## Artifacts

- `artifacts/recommend_hybrid/counterfactual/full_evaluation_input_registry.json`
- `artifacts/recommend_hybrid/counterfactual/full_cohort/progress.json`
- `artifacts/recommend_hybrid/counterfactual/full_cohort/batch_registry.json`
- `artifacts/recommend_hybrid/counterfactual/full_cohort/evaluation_rows.parquet`
- `artifacts/recommend_hybrid/counterfactual/full_cohort/action_scores.parquet`
- `artifacts/recommend_hybrid/counterfactual/full_cohort/evaluation.json`
- `artifacts/recommend_hybrid/counterfactual/full_cohort/bootstrap.json`
- `artifacts/recommend_hybrid/counterfactual/full_cohort/CHECKSUMS.json`
- `artifacts/recommend_hybrid/counterfactual/full_cohort/success_metric_audit.json`
- `artifacts/recommend_hybrid/counterfactual/full_cohort/coverage_analysis.json`
- `artifacts/recommend_hybrid/counterfactual/full_cohort/stability_analysis.json`
- `artifacts/recommend_hybrid/counterfactual/full_cohort/baseline_comparison.json`
- `artifacts/recommend_hybrid/counterfactual/full_cohort/fallback_rows.parquet`
- `artifacts/recommend_hybrid/counterfactual/CANDIDATE_RELEASE_REGISTRY.json`
- `artifacts/recommend_hybrid/expert_review/EXPERT_REVIEW_CASES.csv`
- `artifacts/recommend_hybrid/expert_review/EXPERT_REVIEW_CASES.json`
- `artifacts/recommend_hybrid/expert_review/EXPERT_REVIEW_RUBRIC.csv`
- `artifacts/recommend_hybrid/expert_review/EXPERT_REVIEW_RESULTS_TEMPLATE.csv`

## Reports và blockers

- `reports/recommend_hybrid/COUNTERFACTUAL_SUCCESS_METRIC_AUDIT.md`
- `reports/recommend_hybrid/COUNTERFACTUAL_COVERAGE_AND_FALLBACK.md`
- `reports/recommend_hybrid/COUNTERFACTUAL_STABILITY_ANALYSIS.md`
- `reports/recommend_hybrid/COUNTERFACTUAL_BASELINE_COMPARISON.md`
- `docs/recommend_hybrid/EXPERT_REVIEW_GUIDE.md`
- `docs/recommend_hybrid/EXPERT_REVIEW_PROTOCOL.md`

Remaining blockers:

1. Expert review chưa hoàn tất.
2. Seed stability mới ở mức descriptive ensemble-only; chưa có bảng single-seed độc lập.
3. Assessment availability fraction không được persist trong evaluator frozen.

Vì vậy candidate vẫn là `CANDIDATE_VALIDATED_PENDING_EXPERT_REVIEW`; không merge và không `FINAL_RELEASE`.
