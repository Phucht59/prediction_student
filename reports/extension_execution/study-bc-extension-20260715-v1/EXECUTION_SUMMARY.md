# Study B + Study C execution summary

## Status

- Study A remained frozen; no official Study A evidence was modified and the 79 `legacy_heldout_observed` records were not accessed.
- Study B independent `student-por`: PASS.
- Study B frozen cross-subject transfer: PASS, with overlap limitation.
- Study C OULAD F1/F2/F3 materialization, grouped development evaluation, and future-presentation evaluation: PASS.
- Deep-learning advantage verdict: **NOT_SUPPORTED**.

## Study B

| candidate_id | accuracy | macro_f1 | macro_pr_auc | class_collapse |
| --- | --- | --- | --- | --- |
| B-RF0 | 0.9014 | 0.8698 | 0.9315 | False |
| B-S0 | 0.8952 | 0.8659 | 0.9308 | False |
| B-H0 | 0.8968 | 0.8628 | 0.9329 | False |
| B-H1 | 0.8752 | 0.8470 | 0.9273 | False |
| B-L0 | 0.8844 | 0.8449 | 0.9326 | False |
| B-R0 | 0.8428 | 0.8166 | nan | False |
| B-L1 | 0.8459 | 0.7958 | 0.8754 | False |
| B-M0 | 0.6934 | 0.4047 | 0.7479 | True |
| B-O0 | 0.2943 | 0.3608 | 0.8098 | True |
| B-C0 | 0.6441 | 0.2612 | 0.6921 | True |

Best ML is **B-RF0** (Macro-F1 0.8698); best DL is **B-H1** (Macro-F1 0.8470). This repeats Study A's qualitative finding that compact deep models do not surpass the strongest ML/reference approach on two late-stage grades.

Frozen transfer on all 649 Portuguese records:

| candidate_id | records | accuracy | macro_f1 |
| --- | --- | --- | --- |
| N0 | 649 | 0.8737 | 0.8445 |
| M1 | 649 | 0.8567 | 0.8250 |
| M2 | 649 | 0.8444 | 0.8181 |
| R0 | 649 | 0.8428 | 0.8166 |

This is a frozen cross-subject transfer evaluation, not independent external validation, because quasi-identity overlap exists between the mathematics and Portuguese datasets.

## Study C

| forecast_id | cohort_size | prevalence | best_ml | best_ml_macro_f1 | best_dl | best_dl_macro_f1 | flagship_macro_f1 | flagship_at_risk_recall | flagship_pr_auc | flagship_minus_best_ml | future_flagship_macro_f1 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| F1_EARLY | 26734 | 0.4247 | C-L0 | 0.7436 | C-H2 | 0.7427 | 0.7427 | 0.6840 | 0.7974 | -0.0009 | 0.6683 |
| F2_MIDDLE | 24603 | 0.3747 | C-L0 | 0.8257 | C-M0 | 0.8247 | 0.8149 | 0.7895 | 0.8856 | -0.0108 | 0.7901 |
| F3_LATE | 23034 | 0.3321 | C-H0 | 0.8792 | C-M0 | 0.8749 | 0.8704 | 0.8293 | 0.9311 | -0.0089 | 0.8427 |

The preregistered flagship C-H2 fails the advantage rule: F2 delta is -0.0108, and its delta is positive in 0/3 development forecasts. The negative result is retained. Future-presentation results are domain-shift evidence and were never used for tuning.

## Validation

- Full suite: 169 passed, 5 skipped, 0 failed (return code 0).
- Student grouping: global `id_student` exclusion between historical development and future test, plus grouped nested folds.
- Event contract: exact `[0, cutoff_day)` filtering before weekly aggregation.
- Target and feature snapshots are physically separate.
- SVM C-S0: `SKIPPED_COMPUTE_GATE_CPU_ONLY_RBF_ON_15K_PLUS_ROWS`; this is not represented as PASS.

## Evidence

- `artifacts/study_b_student_por/study-b-student-por-20260715-v1/`
- `reports/study_b_student_por/study-b-student-por-20260715-v1/`
- `artifacts/study_c_oulad/study-c-oulad-20260715-v1/`
- `reports/study_c_oulad/study-c-oulad-20260715-v1/`
