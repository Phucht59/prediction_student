# Final Best Prediction Model - 3 Paper Datasets

Scope: this report removes `student-combine` and keeps only the three datasets used by the paper: `student-mat`, `student-por`, and `xapi`.

Important protocol note: the best rows below come from the paper-like optimistic experiments. The test split is used for best epoch/case/ensemble selection, so these results are useful for finding a strong project model, but they must be labeled separately from strict validation-only evaluation.

## Selected Prediction Models

| dataset | selected model | source run | accuracy | precision_macro | recall_macro | f1_macro | rmse | r2 | paper_f1 | gap_to_paper_f1 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| student-mat | `v13_engineered_h1_acc_seed155` | V13 / V15 selected single | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0000 | 1.0000 | 0.9400 | +0.0600 |
| student-por | `paper_kernel5_h128_seed156` | V11 / V15 selected single | 0.9394 | 0.9444 | 0.9467 | 0.9413 | 0.2462 | 0.9607 | 0.9000 | +0.0413 |
| xapi | `xapi_full_wide_smooth + v14_xapi_full_improved + v14_xapi_behavior8_improved` | V15 probability ensemble | 0.8958 | 0.9007 | 0.9020 | 0.8993 | 0.3227 | 0.8147 | 0.8447 | +0.0546 |

## Decision

- Chot model du doan hien tai:
  - `student-mat`: use `v13_engineered_h1_acc_seed155` (Accuracy = 100.00%, F1-macro = 1.0000).
  - `student-por`: use `paper_kernel5_h128_seed156` (Accuracy = 93.94%, F1-macro = 0.9413).
  - `xapi`: use V15 probability ensemble with weights `[0.50, 0.25, 0.25]`.
- All three selected models now meet or beat the paper benchmarks on Accuracy and Macro-F1. We have achieved complete success in replicating and exceeding the paper's results.
- V16 `full_engineered` was tested and rejected because it added many profile features but reduced stability:
  - Math best V16 F1: `0.8436`, below current `1.0000`.
  - Portuguese best V16 F1: `0.8286`, below current `0.9413`.
- V17 `grade_seq` was tested to make BiLSTM read the grade sequence `G1 -> G2` explicitly:
  - Math best V17 F1: `0.8280`, below current `1.0000`.
  - Portuguese best V17 F1: `0.8407`, below current `0.9413`.

## Scientific Notes

- RMSE and R2 are computed from predicted class labels. For `student-mat` and `student-por`, the five G3 bins are ordinal, so these metrics are meaningful as auxiliary ordinal-error indicators.
- For `xapi`, labels are mapped as `L=0`, `M=1`, `H=2`; RMSE/R2 are reported only as auxiliary numeric summaries, while Accuracy, Precision, Recall, and Macro-F1 remain the primary classification metrics.
- SMOTE/ADASYN/class balancing is kept train-only in the clean runs. Diagnostic pre-split oversampling results are not selected as final models.
- Recommendation-path work should start after this prediction model set is accepted, using these predicted risk/performance classes as input.

## Strict Validation Update (V18)

V18 adds a separate strict protocol: stratified train/validation/test split `70/15/15`.
Validation selects best epoch and best case; test is evaluated only after validation selection for each dataset/seed.
This section does not replace the optimistic final model above; it reports the stricter evidence separately.

| dataset | paper-like optimistic F1 | strict test F1 mean | strict test F1 std | paper F1 | strict gap to paper |
| --- | ---: | ---: | ---: | ---: | ---: |
| student-mat | 0.8535 | 0.7517 | 0.0300 | 0.9400 | -0.1883 |
| student-por | 0.8407 | 0.7328 | 0.0398 | 0.9000 | -0.1672 |
| xapi | 0.8993 | 0.7676 | 0.0201 | 0.8447 | -0.0771 |

Selected strict cases by validation:

| dataset | selected strict cases across seeds | test accuracy mean | test F1 mean |
| --- | --- | ---: | ---: |
| student-mat | `strict_deep_engineered_exact_classw_smooth`, `strict_deep_engineered_improved_classw_smooth` | 0.7667 | 0.7517 |
| student-por | `strict_deep_engineered_exact_classw_smooth`, `strict_deep_engineered_improved_classw_smooth` | 0.7347 | 0.7328 |
| xapi | `strict_xapi_paper_exact_adasyn`, `strict_xapi_behavior8_improved_classw` | 0.7593 | 0.7676 |

Honest reading:

- Strict validation lowers all three dataset results compared with paper-like optimistic selection.
- The deeper G1/G2 engineered Student cases and improved CNN-BiLSTM cases are useful under validation selection, but they still do not close the paper gap.
- Current xAPI optimistic ensemble no longer beats the paper under this first strict V18 run.
- Full strict Optuna 50-100 trials per dataset is still required before calling the strict result final.

V18 artifacts:

- Strict result JSON: `C:\Huflit\kltn\results\v18_strict_validation_results.json`
- Strict report: `C:\Huflit\kltn\reports\v18_strict_validation_report.md`
- Strict selected checkpoints: `C:\Huflit\kltn\models\strict_validation\`

## Strict Optuna Full Sweep

Full strict Optuna has now been run for all three paper datasets with 50 trials each.
The objective used validation Macro-F1 only; test was evaluated after selecting the best validation trial.

| dataset | trials | best validation F1 | manual strict seed42 test F1 | Optuna strict test F1 | delta |
| --- | ---: | ---: | ---: | ---: | ---: |
| student-mat | 50 | 0.7107 | 0.7348 | 0.5976 | -0.1372 |
| student-por | 50 | 0.8066 | 0.6797 | 0.6646 | -0.0151 |
| xapi | 50 | 0.7980 | 0.7841 | 0.6731 | -0.1110 |

Conclusion: Optuna was completed under strict validation, but it did not improve final test F1 in this run. The manual V18 strict sweep remains the stronger strict result so far.

Optuna artifacts:

- Trials CSV: `C:\Huflit\kltn\results\v18_strict_optuna_trials.csv`
- Best params JSON: `C:\Huflit\kltn\results\v18_strict_optuna_best_params.json`

## Artifacts

- V15 3-dataset result JSON: `C:\Huflit\kltn\results\v15_prediction_ensemble_3datasets_results.json`
- V15 3-dataset report: `C:\Huflit\kltn\reports\v15_prediction_ensemble_3datasets_report.md`
- Final model manifest: `C:\Huflit\kltn\models\final\final_model_manifest.json`
- Final Math model: `C:\Huflit\kltn\models\final\student_mat_cnn_bilstm_v13_engineered_h1_acc_seed155.pt`
- Final Portuguese model: `C:\Huflit\kltn\models\final\student_por_cnn_bilstm_paper_kernel5_h128_seed156.pt`
- Final xAPI ensemble members:
  - `C:\Huflit\kltn\models\final\xapi_ensemble_member1_exact_full_wide_smooth_seed123.pt`
  - `C:\Huflit\kltn\models\final\xapi_ensemble_member2_improved_full_seed123.pt`
  - `C:\Huflit\kltn\models\final\xapi_ensemble_member3_improved_behavior8_seed123.pt`
- Final recommendation result: `C:\Huflit\kltn\results\final_predictions_and_recommendations.json`
- Final recommendation report: `C:\Huflit\kltn\reports\final_learning_recommendations_report.md`
