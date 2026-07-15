# Student Performance Prediction and Governed Learning-Path Recommendation

## 1. Project overview

Repository này nghiên cứu dự đoán mức kết quả cuối kỳ từ điểm `G1`, `G2` và xây dựng hệ thống khuyến nghị lộ trình học có quản trị. Hệ thống gồm hai lớp độc lập: prediction được freeze từ development evidence và recommendation dựa trên luật chuyên gia, luôn cần advisor review.

Trạng thái cuối:

- `final_overall_model`: **R0 — G2 deterministic rule**.
- `final_thesis_hybrid_model`: **N0 — nominal CNN–BiLSTM five-seed ensemble**.
- Recommendation: `technical_validation = PASS`, `expert_validation = PENDING`, `effectiveness_validation = NOT_PERFORMED`.

## 2. Thesis objectives

1. So sánh minh bạch các baseline ML và kiến trúc CNN–BiLSTM trên cùng hợp đồng G1/G2.
2. Đánh giá classification, ordinal error, calibration, stability và continuous-G3 analysis mà không tạo composite score hậu nghiệm.
3. Giữ CNN–BiLSTM như kiến trúc nghiên cứu của khóa luận, không ép kiến trúc này thành overall champion.
4. Sinh draft lộ trình bốn tuần có mục tiêu, hành động, giải thích, advisor decision, follow-up và immutable revision.

## 3. Scientific scope

Đây là nghiên cứu **development-selected and development-frozen**. Không có tập xác nhận ngoài hoàn toàn chưa từng quan sát. Hệ thống recommendation là **expert-guided, rule-based, human-in-the-loop và non-causal**; nó không phải learned recommender, causal recommender hay reinforcement-learning recommender.

Không được dùng repository để tuyên bố CNN–BiLSTM vượt trội ML, mô hình đã chứng minh khả năng tổng quát hóa thực tế, hoặc khuyến nghị đã làm tăng điểm.

## 4. Dataset and target definition

- Nguồn: UCI Student Performance, bảng `student-mat`.
- Tổng số: 395 records.
- Development protocol: 316 records.
- 79 records còn lại có trạng thái `legacy_heldout_observed`; chúng đã bị quan sát trong lịch sử và không còn là locked test hợp lệ.
- Input prediction: `G1`, `G2`.
- Target: raw `G3`, chỉ nằm trong target storage/evaluation contract.
- Bins: Low = G3 0–9, Medium = 10–14, High = 15–20; class order 0/1/2.

`G2-G1` chỉ là deterministic trajectory dùng cho explanation/planning, không đi vào frozen prediction model.

## 5. Data lineage and PostgreSQL architecture

PostgreSQL tách source records, target records, split membership, predictions và governed recommendation lineage. Migrations nằm trong [`database/migrations`](database/migrations):

- `001`: source/ML schema;
- `002`: append-only recommendation policy versions;
- `003`: tách target khỏi source payload;
- `004`: policy, feature/action registries, prediction snapshots, immutable revisions, advisor decisions, follow-ups, outcomes và expert review.

Official Phase D chỉ đọc G1/G2 bằng development source-row allowlist trong read-only transaction; không join target table và không fetch 79 observed.

## 6. Validation protocol

- 5 immutable outer folds × 3 inner folds trên 316 development records.
- Phase C: tối đa 30 Optuna trials/family, cùng fold/feature/training contracts.
- Phase E stability seeds: `202601`, `202602`, `202603`, `202604`, `202605`; không chọn best seed.
- Macro-F1 là primary classification metric.
- Accuracy, Precision, Recall, F1 và PR-AUC là classification metrics.
- RMSE/R² là continuous-G3 secondary analysis; R0 dùng `predicted_G3 = G2`, còn classification-only models dùng class-conditional training-partition means.
- Không trộn classification và regression metrics thành composite score.
- Calibration chỉ được đánh giá từ inner-OOF; N0 calibration bị reject và giữ uncalibrated.

## 7. Final model roles

| Model | Role | Final status |
| --- | --- | --- |
| R0 — G2 deterministic rule | Overall development-selected model, sanity/reference guardrail | Frozen |
| M1 — Random Forest | Practical-tie ML comparator; highest point-estimate Macro-F1 | Comparator |
| M2 — SVM RBF | Practical-tie ML comparator | Comparator |
| N0 — nominal CNN–BiLSTM | Thesis hybrid five-seed ensemble and recommendation score source | Frozen |
| N1 — ordinal CNN–BiLSTM | Ordinal research comparator | Comparator |

R0 không có probabilistic uncertainty. N0 outputs được gọi là `model_score`/`ensemble_score`, không phải xác suất đúng tuyệt đối.

## 8. Official development results

| Model | Role | Accuracy | Macro Precision | Macro Recall | Macro-F1 | Weighted-F1 | High-class F1 | Macro PR-AUC | RMSE G3 | R² G3 | Validation scope |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| R0 | final overall model | 0.8924 | 0.9078 | 0.8935 | 0.8988 | 0.8925 | 0.9246 | 0.8461 | 2.0086 | 0.8050 | Nested development OOF |
| M1 | practical-tie ML comparator | 0.8924 | 0.9079 | 0.8924 | **0.9000** | 0.8920 | 0.9332 | 0.9526 | 2.4609 | 0.7065 | Nested development OOF/new-seed stability |
| M2 | practical-tie ML comparator | 0.8829 | 0.9035 | 0.8798 | 0.8901 | 0.8829 | 0.9246 | **0.9602** | 2.3605 | 0.7305 | Nested development OOF/deterministic replay |
| N0 | final thesis hybrid model | 0.8462 | 0.8606 | 0.8535 | 0.8504 | 0.8450 | 0.8694 | 0.9510 | 2.4632 | 0.7067 | Nested development OOF/five new seeds |
| N1 | ordinal research comparator | 0.8315 | 0.8435 | 0.8621 | 0.8383 | 0.8289 | 0.8701 | 0.9457 | 2.4329 | 0.7128 | Nested development OOF/five new seeds |

Random Forest có point-estimate Macro-F1 cao nhất. Tuy nhiên R0, M1 và M2 nằm trong practical tie theo rule đã đăng ký; R0 được chọn bởi tie-break và simplicity, không phải vì superiority thống kê. N0 là thesis hybrid; N0/N1 không có superiority rõ.

## 9. CNN–BiLSTM architecture

N0 dùng `[G1,G2] → compact Conv1D → activation → optional LayerNorm → dropout → compact BiLSTM → nominal three-class logits`. Main contract không dùng BatchNorm, `drop_last=False`, full-record coverage, fixed/replayable LR policy, không SWA và full-fold refit. Final bundle là arithmetic mean của năm checkpoints Phase E.

## 10. Ablation findings

- CNN–BiLSTM đã cải thiện so với estimator CNN–BiLSTM lịch sử đã quarantine, nhưng vẫn thấp hơn các ML baselines chính.
- A2 BiLSTM-only practical-tie với N0 trong Phase C.
- CNN chưa chứng minh incremental value rõ trên chuỗi chỉ hai timestep.
- Ordinal learning chưa chứng minh cải thiện; N1 không vượt N0 rõ ràng.
- Residual, multitask và imbalance branches không được mở vì gates không đạt.

## 11. Recommendation architecture

```text
Frozen N0 five-seed ensemble + R0 agreement guardrail
→ prediction evidence snapshot
→ uncertainty/agreement and feature-governance assessment
→ governed rule-based four-week learning plan
→ advisor approve/modify/reject/request-more-information
→ follow-up and immutable revision
```

Tên mô tả chính thức: **Hệ thống khuyến nghị lộ trình học dựa trên luật chuyên gia, có quản trị, giải thích, advisor review, follow-up và revision.** Không recommendation nào tự động active.

## 12. Recommendation technical validation

| Item | Result |
| --- | ---: |
| Development recommendation cases | 316 |
| Eligible for normal draft gate | 245 |
| Uncertainty/agreement review cases | 71 |
| Gate review rate | 22.47% |
| Drafts requiring advisor approval | 100% |
| Generated actions | 1,313 |
| Action conflicts / duplicates / workload violations | 0 / 0 / 0 |
| Goal / action / explanation completeness | 100% / 100% / 100% |
| Expert casebook | 60 cases / 23 strata |

`technical_validation = PASS`; `expert_validation = PENDING`; `effectiveness_validation = NOT PERFORMED`. Structural correctness không phải bằng chứng recommendation effectiveness.

## 13. Reproducibility

Official evidence roots:

- Phase A–B: [`artifacts/strategy_b_phase_ab/strategy-b-phase-ab-20260714-475a672`](artifacts/strategy_b_phase_ab/strategy-b-phase-ab-20260714-475a672)
- Phase C: [`artifacts/strategy_b_phase_c/strategy-b-phase-c-20260714-5d34a66`](artifacts/strategy_b_phase_c/strategy-b-phase-c-20260714-5d34a66)
- Phase E: [`artifacts/strategy_b_phase_e_prediction/strategy-b-phase-e-prediction-20260714-9007144`](artifacts/strategy_b_phase_e_prediction/strategy-b-phase-e-prediction-20260714-9007144)
- Phase D: [`artifacts/strategy_b_phase_d_recommendation/strategy-b-phase-d-recommendation-20260715-407ac0f`](artifacts/strategy_b_phase_d_recommendation/strategy-b-phase-d-recommendation-20260715-407ac0f)

Artifact checksum manifests và source provenance là nguồn kiểm tra; không cần retrain để xác minh repository.

## 14. Repository structure

```text
config/       feature availability contracts
database/     PostgreSQL migrations
src/          estimator, models, metrics, lineage and governed policy
scripts/      ingestion, historical experiment runners and validators
tests/        unit/contract/integration tests
artifacts/    immutable machine-readable evidence
reports/      human-readable evidence mirrors and thesis context
docs/         historical design/audit documents; registry controls headline use
```

## 15. How to run

Environment:

```powershell
python -m pip install -r requirements-lock.txt
Copy-Item .env.example .env
```

PostgreSQL ingestion (mutating; only on an authorized database):

```powershell
python scripts/ingest_dataset_to_postgres.py --dataset student-mat
```

Database migrations must be applied in numeric order with `psql -v ON_ERROR_STOP=1 -f <migration>`. Migration 004 destructive tests require disposable `POSTGRES_TEST_DSN`; do not use production.

Quick portable validation:

```powershell
python scripts/verify_final_evidence.py --skip-db
python -m pytest -q
```

Official Phase validators are checksum/strict artifacts: inspect `strict_validation.json` and verify `artifact_checksums.json` under the Phase A–B/C/E/D roots above. Phase C/E runners and Optuna commands are expensive historical reproduction entrypoints, not quick validation commands.

## 16. Tests

Run the exact current suite with:

```powershell
python -m pytest -q -rs
```

Five PostgreSQL destructive integration tests may skip when disposable DSNs/`psql` are unavailable. A skip is not a pass and production DB must not be used to satisfy them. The authoritative closure count is stored in the newest `artifacts/final_repository_closure/<run_id>/test_report.json`, never hard-coded here.

## 17. Evidence registry

The closure bundle provides `official_evidence_registry.json`, `historical_evidence_registry.json` and `artifact_index.json`. Headline-eligible evidence is limited to Phase C main comparison, Phase E final development freeze and Phase D technical validation.

Historical locked-test results, the 79 observed records, old fair-DL rows with resolved-config mismatch, smoke runs, residual diagnostics and old-estimator CNN–BiLSTM results are not headline evidence. They remain preserved for auditability.

## 18. Limitations

- Small dataset (395 total; 316 usable development records).
- Sequence length is two, limiting claims about temporal modeling.
- No untouched external confirmation dataset.
- The 79 historical held-out records were observed and are contaminated for confirmation.
- N0 is uncalibrated; R0 has no uncertainty estimate.
- Context features are not activated in the governed recommendation policy.
- Expert review and prospective effectiveness evaluation remain outstanding.

## 19. Ethical and scientific boundaries

- No automatic educational decision, ranking, discipline or denial of support.
- No causal interpretation of feature associations or recommendation actions.
- No fake confidence for deterministic R0.
- Every recommendation remains a draft until advisor review.
- Sensitive/non-actionable attributes are excluded from active recommendation rules.
- Future deployment requires data minimization, access control, audit logs, monitoring and local validation.

## 20. Thesis-writing sources

Use the newest closure files:

- `reports/final_repository_closure/<run_id>/thesis_writing_context.md`
- `reports/final_repository_closure/<run_id>/thesis_evidence_map.csv`
- Phase E `stability_summary.csv`, `confusion_matrices.csv`, `precision_recall_curve_points.csv`, `paired_stability_deltas.csv` and `final_model_manifest.json`.
- Phase D `technical_safety_metrics.json`, `coverage_and_abstention.csv`, `expert_casebook.csv` and `strict_validation.json`.

These sources support thesis writing; no DOCX is generated or edited by repository closure.
