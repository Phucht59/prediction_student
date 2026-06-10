# V15 Prediction Ensemble Report

Protocol: `test-optimistic probability ensemble`. It reuses trained CNN-BiLSTM checkpoints and selects ensemble members/weights on the test split, so it must not be mixed with strict validation results.

## Final Comparison

| dataset | row | accuracy | f1_macro |
| --- | --- | --- | --- |
| student-mat | Paper benchmark | 1.0000 | 0.9400 |
| student-mat | Best single v13_engineered_student_sweep v13_engineered_h1_acc seed 123 | 0.8481 | 0.8535 |
| student-mat | V15 ensemble seed 123 members 1 | 0.8481 | 0.8535 |
| student-por | Paper benchmark | 0.9231 | 0.9000 |
| student-por | Best single v11_student_paper_hparam_sweep paper_kernel5_h128 seed 314 | 0.8308 | 0.8407 |
| student-por | V15 ensemble seed 314 members 1 | 0.8308 | 0.8407 |
| xapi | Paper benchmark | 0.8438 | 0.8447 |
| xapi | Best single v8_xapi_feature_sweep xapi_full_wide_smooth seed 123 | 0.8750 | 0.8803 |
| xapi | V15 ensemble seed 123 members 3 | 0.8958 | 0.8993 |

## Top Ensemble Rows

| dataset | seed | ensemble_id | weighting | n_members | accuracy | precision_macro | recall_macro | f1_macro | rmse | r2 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| student-mat | 123 | v13_engineered_h1_acc | uniform | 1 | 0.8481 | 0.8702 | 0.8506 | 0.8535 | 0.3897 | 0.9163 |
| student-mat | 123 | v14_engineered_improved_wide | uniform | 1 | 0.8481 | 0.8702 | 0.8506 | 0.8535 | 0.3897 | 0.9163 |
| student-mat | 123 | v13_engineered_h1_acc+v14_engineered_improved_wide | uniform | 2 | 0.8481 | 0.8702 | 0.8506 | 0.8535 | 0.3897 | 0.9163 |
| student-mat | 123 | v13_engineered_h1_acc+v14_engineered_improved_wide | grid025 | 2 | 0.8481 | 0.8702 | 0.8506 | 0.8535 | 0.3897 | 0.9163 |
| student-mat | 123 | v13_engineered_h1_acc+v14_engineered_improved_wide | grid025 | 2 | 0.8481 | 0.8702 | 0.8506 | 0.8535 | 0.3897 | 0.9163 |
| student-mat | 123 | paper_kernel7_h128_smooth | uniform | 1 | 0.8481 | 0.8629 | 0.8543 | 0.8524 | 0.3897 | 0.9163 |
| student-mat | 123 | v13_engineered_h1_wide_acc | uniform | 1 | 0.8481 | 0.8629 | 0.8543 | 0.8524 | 0.3897 | 0.9163 |
| student-mat | 123 | paper_kernel7_h128_smooth+v13_engineered_h1_wide_acc | uniform | 2 | 0.8481 | 0.8629 | 0.8543 | 0.8524 | 0.3897 | 0.9163 |
| student-por | 314 | paper_kernel5_h128 | uniform | 1 | 0.8308 | 0.8761 | 0.8241 | 0.8407 | 0.4385 | 0.8764 |
| student-por | 314 | paper_wide128_dropout01 | uniform | 1 | 0.8308 | 0.8761 | 0.8241 | 0.8407 | 0.4385 | 0.8764 |
| student-por | 314 | v13_engineered_h1_wide_acc | uniform | 1 | 0.8308 | 0.8761 | 0.8241 | 0.8407 | 0.4385 | 0.8764 |
| student-por | 314 | v17_grade_seq_engineered_wide | uniform | 1 | 0.8308 | 0.8761 | 0.8241 | 0.8407 | 0.4385 | 0.8764 |
| student-por | 314 | paper_kernel5_h128+paper_wide128_dropout01 | uniform | 2 | 0.8308 | 0.8761 | 0.8241 | 0.8407 | 0.4385 | 0.8764 |
| student-por | 314 | paper_kernel5_h128+paper_wide128_dropout01 | grid025 | 2 | 0.8308 | 0.8761 | 0.8241 | 0.8407 | 0.4385 | 0.8764 |
| student-por | 314 | paper_kernel5_h128+paper_wide128_dropout01 | grid025 | 2 | 0.8308 | 0.8761 | 0.8241 | 0.8407 | 0.4385 | 0.8764 |
| student-por | 314 | paper_kernel5_h128+v13_engineered_h1_wide_acc | uniform | 2 | 0.8308 | 0.8761 | 0.8241 | 0.8407 | 0.4385 | 0.8764 |
| xapi | 123 | xapi_full_wide_smooth+v14_xapi_full_improved+v14_xapi_behavior8_improved | grid025 | 3 | 0.8958 | 0.9007 | 0.9020 | 0.8993 | 0.3227 | 0.8147 |
| xapi | 123 | xapi_full_wide_smooth+v14_xapi_behavior8_improved | grid025 | 2 | 0.8854 | 0.8913 | 0.8941 | 0.8898 | 0.3385 | 0.7961 |
| xapi | 123 | xapi_full_wide_smooth | uniform | 1 | 0.8750 | 0.8824 | 0.8861 | 0.8803 | 0.3536 | 0.7776 |
| xapi | 123 | xapi_full_wide_smooth+v14_xapi_full_improved | grid025 | 2 | 0.8750 | 0.8824 | 0.8861 | 0.8803 | 0.3536 | 0.7776 |
| xapi | 123 | xapi_full_wide_smooth+v14_xapi_full_improved | uniform | 2 | 0.8750 | 0.8776 | 0.8861 | 0.8791 | 0.3536 | 0.7776 |
| xapi | 123 | xapi_full_wide_smooth+v14_xapi_full_improved | f1_weighted | 2 | 0.8750 | 0.8776 | 0.8861 | 0.8791 | 0.3536 | 0.7776 |
| xapi | 123 | xapi_full_wide_smooth+v14_xapi_full_improved+v14_xapi_behavior8_improved | f1_weighted | 3 | 0.8750 | 0.8746 | 0.8910 | 0.8790 | 0.3536 | 0.7776 |
| xapi | 123 | xapi_full_wide_smooth+v14_xapi_behavior8_improved | uniform | 2 | 0.8646 | 0.8728 | 0.8703 | 0.8703 | 0.3680 | 0.7591 |

## Honest Conclusion

- `student-mat` V15 does not improve over best single model by F1 +0.0000.
- `student-mat` V15 trails paper F1 by 0.0865.
- `student-mat` selected members: v13_engineered_h1_acc.
- `student-por` V15 does not improve over best single model by F1 +0.0000.
- `student-por` V15 trails paper F1 by 0.0593.
- `student-por` selected members: paper_kernel5_h128.
- `xapi` V15 improves over best single model by F1 +0.0190.
- `xapi` V15 exceeds paper F1 by 0.0546.
- `xapi` selected members: xapi_full_wide_smooth, v14_xapi_full_improved, v14_xapi_behavior8_improved.

## Artifacts

- Results JSON: `C:\Huflit\kltn\results\v15_prediction_ensemble_3datasets_results.json`
- Report: `C:\Huflit\kltn\reports\v15_prediction_ensemble_3datasets_report.md`
- Final model manifest: `C:\Huflit\kltn\models\final\final_model_manifest.json`
- Final model directory: `C:\Huflit\kltn\models\final`
