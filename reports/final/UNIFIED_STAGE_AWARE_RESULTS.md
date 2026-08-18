# Unified Stage-Aware UCI Results

One estimator is fitted per dataset/model/fold/seed and reused for S0, S1, and S2. A stage is a prediction view, not a model identity. Outer rows are never used for configuration selection.

## Stage results

| Dataset | Stage | Best model | Macro-F1 | Low recall |
|---|---|---|---:|---:|
| student_mat | S0_EARLY_NO_GRADE | svm_mat | 0.4523 | 0.4385 |
| student_mat | S1_MID_G1_ONLY | cnn_only_mat | 0.7522 | 0.8154 |
| student_mat | S2_LATE_G1_G2 | random_forest_mat | 0.8893 | 0.8769 |
| student_por | S0_EARLY_NO_GRADE | bilstm_only_por | 0.5094 | 0.6600 |
| student_por | S1_MID_G1_ONLY | random_forest_por | 0.7835 | 0.8000 |
| student_por | S2_LATE_G1_G2 | random_forest_por | 0.8571 | 0.8400 |

## Claim boundaries

- UCI results are the unified fixed-fold authority produced by this branch.
- S2 is the late-stage UCI prediction view; S0 and S1 quantify earlier availability.
- OULAD remains frozen at `F2_MIDDLE`; it was not retrained.
- The grade-band reference is training-fold-only diagnostic evidence and is not an eleventh model.
- Future OULAD remains `LOCKED_NOT_EXECUTED`.

Validation state at report generation: `PASS`.
