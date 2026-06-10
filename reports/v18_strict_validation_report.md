# V18 Strict Validation Report

Protocol: train/validation/test = 70/15/15. Validation selects epoch and case. Test is evaluated only after validation selection for each dataset/seed.

## Paper-like Optimistic vs Strict

| dataset | paper_like_optimistic_f1 | strict_test_f1_mean | strict_test_f1_std | paper_f1 | strict_gap_to_paper |
| --- | --- | --- | --- | --- | --- |
| student-mat | 0.8535 | 0.7517 | 0.0300 | 0.9400 | -0.1883 |
| student-por | 0.8407 | 0.7328 | 0.0398 | 0.9000 | -0.1672 |
| xapi | 0.8993 | 0.7676 | 0.0201 | 0.8447 | -0.0771 |

## Strict Test Summary

| dataset | n_seeds | selected_cases | val_f1_macro_mean | val_f1_macro_std | test_accuracy_mean | test_accuracy_std | test_f1_macro_mean | test_f1_macro_std | gap_to_paper_f1 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| student-mat | 3 | strict_deep_engineered_exact_classw_smooth, strict_deep_engineered_improved_classw_smooth | 0.7846 | 0.0795 | 0.7667 | 0.0236 | 0.7517 | 0.0300 | -0.1883 |
| student-por | 3 | strict_deep_engineered_exact_classw_smooth, strict_deep_engineered_improved_classw_smooth | 0.7765 | 0.0230 | 0.7347 | 0.0382 | 0.7328 | 0.0398 | -0.1672 |
| xapi | 3 | strict_xapi_behavior8_improved_classw, strict_xapi_paper_exact_adasyn | 0.8048 | 0.0326 | 0.7593 | 0.0262 | 0.7676 | 0.0201 | -0.0771 |

## Selected Rows

| dataset | seed | case | best_epoch | val_f1_macro | test_accuracy | test_f1_macro | model_path |
| --- | --- | --- | --- | --- | --- | --- | --- |
| student-mat | 42 | strict_deep_engineered_exact_classw_smooth | 15 | 0.6731 | 0.7500 | 0.7348 | C:\Huflit\kltn\models\strict_validation\v18_strict_student-mat_strict_deep_engineered_exact_classw_smooth_seed42.pt |
| student-mat | 123 | strict_deep_engineered_improved_classw_smooth | 12 | 0.8532 | 0.8000 | 0.7938 | C:\Huflit\kltn\models\strict_validation\v18_strict_student-mat_strict_deep_engineered_improved_classw_smooth_seed123.pt |
| student-mat | 314 | strict_deep_engineered_improved_classw_smooth | 15 | 0.8275 | 0.7500 | 0.7265 | C:\Huflit\kltn\models\strict_validation\v18_strict_student-mat_strict_deep_engineered_improved_classw_smooth_seed314.pt |
| student-por | 42 | strict_deep_engineered_exact_classw_smooth | 14 | 0.7441 | 0.6837 | 0.6797 | C:\Huflit\kltn\models\strict_validation\v18_strict_student-por_strict_deep_engineered_exact_classw_smooth_seed42.pt |
| student-por | 123 | strict_deep_engineered_improved_classw_smooth | 16 | 0.7920 | 0.7449 | 0.7433 | C:\Huflit\kltn\models\strict_validation\v18_strict_student-por_strict_deep_engineered_improved_classw_smooth_seed123.pt |
| student-por | 314 | strict_deep_engineered_improved_classw_smooth | 13 | 0.7935 | 0.7755 | 0.7754 | C:\Huflit\kltn\models\strict_validation\v18_strict_student-por_strict_deep_engineered_improved_classw_smooth_seed314.pt |
| xapi | 42 | strict_xapi_paper_exact_adasyn | 12 | 0.7587 | 0.7778 | 0.7841 | C:\Huflit\kltn\models\strict_validation\v18_strict_xapi_strict_xapi_paper_exact_adasyn_seed42.pt |
| xapi | 123 | strict_xapi_behavior8_improved_classw | 11 | 0.8261 | 0.7222 | 0.7392 | C:\Huflit\kltn\models\strict_validation\v18_strict_xapi_strict_xapi_behavior8_improved_classw_seed123.pt |
| xapi | 314 | strict_xapi_behavior8_improved_classw | 19 | 0.8295 | 0.7778 | 0.7794 | C:\Huflit\kltn\models\strict_validation\v18_strict_xapi_strict_xapi_behavior8_improved_classw_seed314.pt |

## Strict Optuna Full Sweep

Optuna was run with strict validation objective only: `validation_macro_f1`.
Each dataset used 50 trials, seed `42`, 40 epochs per trial, then retrained the selected best trial for 60 epochs before one final test evaluation.

| dataset | trials | best trial | best validation F1 | manual strict seed42 test F1 | Optuna strict test F1 | delta vs manual seed42 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| student-mat | 50 | 1 | 0.7107 | 0.7348 | 0.5976 | -0.1372 |
| student-por | 50 | 41 | 0.8066 | 0.6797 | 0.6646 | -0.0151 |
| xapi | 50 | 45 | 0.7980 | 0.7841 | 0.6731 | -0.1110 |

Best Optuna configurations:

| dataset | feature_set | model_variant | loss | oversampling | key params | checkpoint |
| --- | --- | --- | --- | --- | --- | --- |
| student-mat | paper | exact | ce | none | conv=128, kernel=5, hidden=96, dense=192, dropout=0.25, lr=0.000252 | C:\Huflit\kltn\models\strict_validation\v18_strict_student-mat_strict_optuna_best_student-mat_seed42.pt |
| student-por | deep_engineered | improved | ce | smote | conv=96, kernel=7, hidden=96, dense=64, dropout=0.15, lr=0.000506 | C:\Huflit\kltn\models\strict_validation\v18_strict_student-por_strict_optuna_best_student-por_seed42.pt |
| xapi | full | exact | class_weight | none effective | conv=64, kernel=3, hidden=64, dense=192, dropout=0.30, lr=0.000727 | C:\Huflit\kltn\models\strict_validation\v18_strict_xapi_strict_optuna_best_xapi_seed42.pt |

Honest Optuna conclusion:

- Full Optuna satisfies the 50-trial requirement for all three datasets, but the selected validation-best configurations did not improve final test F1 in this run.
- The gap indicates validation overfitting/model-selection variance on small datasets; the strict manual sweep remains the stronger strict result so far.
- The Optuna artifacts are kept for reproducibility: `results/v18_strict_optuna_trials.csv` and `results/v18_strict_optuna_best_params.json`.

## Notes

- Rows in `validation_rows` do not contain test metrics unless they are selected by validation.
- This report intentionally separates strict validation metrics from the previous paper-like optimistic manifest.
