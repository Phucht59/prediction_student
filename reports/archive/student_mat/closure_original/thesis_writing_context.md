# Thesis-writing context

## 1. Đề tài

Đề tài xây dựng hệ thống dự đoán ba mức kết quả cuối kỳ và hệ thống khuyến nghị lộ trình học có quản trị. Phần prediction so sánh baseline ML với CNN–BiLSTM trên cùng G1/G2; phần recommendation biến prediction evidence thành draft mục tiêu/hành động luôn cần advisor review. Phạm vi là development-only, không phải production hoặc causal intervention.

## 2. Dữ liệu

UCI Student Performance `student-mat` có 395 records. Official protocol chỉ dùng 316 development records. 79 records còn lại đã bị quan sát trong lịch sử, mang nhãn `legacy_heldout_observed`, không dùng cho selection/calibration/confirmation. Inputs là G1/G2; raw G3 là target. Bins: Low 0–9, Medium 10–14, High 15–20.

## 3. Kiến trúc

- R0: deterministic thresholds trên G2; final overall model và agreement guardrail.
- M1/M2: Random Forest và SVM RBF practical-tie comparators.
- N0: compact nominal CNN–BiLSTM, five-seed ensemble; final thesis hybrid.
- N1: ordered ordinal CNN–BiLSTM comparator.
- Ablations: tiny MLP, ordered MLP, CNN-only, BiLSTM-only trong Phase C.
- Recommendation: N0 scores + R0 agreement → snapshot → uncertainty/feature governance → rule-based four-week goals/actions → advisor decision → follow-up/revision.

## 4. Protocol

Năm immutable outer folds, ba inner folds, development-only nested selection, replayable estimator/refit contract và Macro-F1 primary. Phase C neural selected configs dùng seeds 42/123/155; Phase E stability dùng new seeds 202601–202605 và không chọn best seed. PR metrics là one-vs-rest. RMSE/R² dùng continuous prediction contracts riêng. Không có locked-test hoặc external-confirmation claim.

## 5. Kết quả prediction

| model | role | accuracy | macro_precision | macro_recall | macro_f1 | weighted_f1 | high_class_f1 | macro_pr_auc | rmse_g3 | r2_g3 | validation_scope |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| R0 | final overall model | 0.8924 | 0.9078 | 0.8935 | 0.8988 | 0.8925 | 0.9246 | 0.8461 | 2.0086 | 0.8050 | nested development OOF; no external confirmation |
| M1 | practical-tie ML comparator | 0.8924 | 0.9079 | 0.8924 | 0.9000 | 0.8920 | 0.9332 | 0.9526 | 2.4609 | 0.7065 | nested development OOF; no external confirmation |
| M2 | practical-tie ML comparator | 0.8829 | 0.9035 | 0.8798 | 0.8901 | 0.8829 | 0.9246 | 0.9602 | 2.3605 | 0.7305 | nested development OOF; no external confirmation |
| N0 | final thesis hybrid model | 0.8462 | 0.8606 | 0.8535 | 0.8504 | 0.8450 | 0.8694 | 0.9510 | 2.4632 | 0.7067 | nested development OOF; no external confirmation |
| N1 | ordinal research comparator | 0.8315 | 0.8435 | 0.8621 | 0.8383 | 0.8289 | 0.8701 | 0.9457 | 2.4329 | 0.7128 | nested development OOF; no external confirmation |

M1 có point Macro-F1 cao nhất, nhưng R0/M1/M2 practical tie. R0 được chọn bởi tie-break/simplicity. N0 là thesis hybrid; N0/N1 không có superiority rõ. N0 calibration bị reject; N1 temperature calibration được giữ cho comparator nhưng không thay đổi final family.

## 6. Kết luận khoa học

ML có lợi thế trên bài toán hai feature và dữ liệu nhỏ. CNN–BiLSTM được giữ để trả lời mục tiêu kiến trúc của khóa luận, không phải overall champion. BiLSTM-only practical-tie với N0; CNN incremental value và ordinal improvement chưa được thiết lập. Residual/multitask/imbalance gates đóng. Recommendation là governed, non-causal và mới chỉ qua technical validation.

Recommendation technical facts: 316 cases; 245 normal-gate cases; 71 uncertainty/agreement review cases (22.47%); 100% require advisor approval; 1313 actions; zero conflict/duplicate/workload violations; 60-case/23-strata expert casebook.

## 7. Hạn chế

Dataset nhỏ; sequence length hai; không có external unseen confirmation; 79 records bị contamination; expert validation pending; effectiveness not performed; context features chưa active; không có prospective intervention study; dataset không đại diện trực tiếp cho sinh viên đại học Việt Nam.

## 8. Figures and tables available

- Architecture: `src/models/phase_c.py`, `artifacts/strategy_b_phase_e_prediction/strategy-b-phase-e-prediction-20260714-9007144/final_model_manifest.json`.
- Metric table/stability: Phase E `stability_summary.csv`, `fold_seed_metrics.csv`.
- Confusion matrices: Phase E `confusion_matrices.csv`.
- PR curves: Phase E `precision_recall_curve_points.csv`, `precision_recall_metrics.csv`.
- Paired comparisons: Phase E `paired_stability_deltas.csv`; Phase C `paired_model_deltas.csv`.
- Calibration: Phase E `calibration_metrics.csv`, `calibration_decision.json`.
- Recommendation flow/policy: Phase D `protocol.json`, `model_role_contract.json`, `action_catalog.json`.
- Database schema: `database/migrations/001_create_source_ml_schema.sql` through `004_governed_recommendation_phase_d.sql`.
- Evidence hierarchy: closure `official_evidence_registry.json`, `historical_evidence_registry.json`, `thesis_evidence_map.csv`.

Use these artifacts to construct figures/tables; this closure does not edit thesis DOCX.
