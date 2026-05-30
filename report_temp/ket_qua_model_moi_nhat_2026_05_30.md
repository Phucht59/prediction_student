# KET QUA MODEL MOI NHAT - 30/05/2026

## 1. Lenh da chay

Da chay toan bo pipeline tu dau vao raw data den bang tong hop ket qua:

```powershell
py scripts\run_all.py
```

Pipeline hoan tat thanh cong luc 18:40 ngay 30/05/2026. Cac buoc da chay:

- Tien xu ly `student-mat`, `student-por`, `student-combined` voi 2 scenario `mid`, `late`.
- Tien xu ly xAPI voi scenario `xapi_behavior`.
- Train baseline classification/regression.
- Train cac thu nghiem imbalance baseline.
- Train deep model `clsv2` cho Student Performance.
- Train deep model `cnn_bilstm_xapi` cho xAPI.
- Tong hop tat ca ket qua vao `reports/tables`.

## 2. Trang thai du lieu

| Dataset | Scenario | Train | Validation | Test | Leakage check |
| --- | --- | ---: | ---: | ---: | --- |
| student-mat | mid | 275 | 60 | 60 | Pass |
| student-mat | late | 275 | 60 | 60 | Pass |
| student-por | mid | 453 | 98 | 98 | Pass |
| student-por | late | 453 | 98 | 98 | Pass |
| student-combined | mid | 730 | 157 | 157 | Pass |
| student-combined | late | 730 | 157 | 157 | Pass |
| xAPI | xapi_behavior | 336 | 72 | 72 | Pass |

Ghi chu: scenario `mid` chi dung G1 va khong dung G2/G3; scenario `late` dung G1/G2 va khong dung G3; xAPI khong dua target `Class` vao feature.

## 3. Ket qua deep model va xu ly mat can bang tot nhat

Bang duoi lay tu `reports/tables/final_imbalance_deep_comparison.csv`, xep hang theo Macro-F1 trong tung dataset/scenario.

| Dataset | Scenario | Model | Strategy tot nhat | Accuracy | Macro-F1 | PR-AUC macro | Recall lop yeu |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: |
| student-combined | late | clsv2 | smote | 0.8471 | 0.8474 | 0.9317 | 0.8857 |
| student-mat | late | clsv2 | class_weight_balanced | 0.8333 | 0.8448 | 0.9260 | 0.8500 |
| student-por | late | clsv2 | class_weight_balanced | 0.8878 | 0.8829 | 0.9585 | 0.8667 |
| xAPI | xapi_behavior | cnn_bilstm_xapi | none | 0.7639 | 0.7708 | 0.7842 | 0.8947 |

Nhan xet nhanh:

- `student-por/late` dat ket qua deep tot nhat trong cac tap Student: Macro-F1 = 0.8829, Accuracy = 0.8878.
- `student-combined/late` tot nhat khi dung SMOTE: Macro-F1 = 0.8474.
- `student-mat/late` tot nhat khi dung class weight balanced: Macro-F1 = 0.8448.
- xAPI deep model tot nhat khi khong oversampling: Macro-F1 = 0.7708, Accuracy = 0.7639.

## 4. So sanh deep voi baseline tabular

Nguon: `reports/tables/deep_vs_tabular_comparison.csv`.

| Dataset | Scenario | Baseline Macro-F1 | Imbalance Macro-F1 | Deep Macro-F1 | Ghi chu |
| --- | --- | ---: | ---: | ---: | --- |
| student-mat | mid | 0.6084 | 0.6084 | 0.7247 | Deep tang +0.1163 so voi baseline |
| student-por | mid | 0.7105 | 0.6971 | 0.7625 | Deep tang +0.0521 so voi baseline |
| student-combined | mid | 0.6957 | 0.7480 | 0.7324 | Deep tot hon baseline, kem imbalance baseline nhe |
| student-por | late | 0.8581 | 0.7963 | 0.8368 | Deep tang recall lop yeu len 0.8667 |
| student-combined | late | 0.8422 | 0.8481 | 0.8317 | Baseline/imbalance nhe hon deep ve Macro-F1 |
| student-mat | late | 0.9372 | 0.9522 | 0.8272 | Tabular baseline dang tot hon deep |

## 5. Ket qua baseline tot nhat

Classification baseline:

| Dataset | Scenario | Model tot nhat | Test Macro-F1 | Accuracy | Recall lop yeu |
| --- | --- | --- | ---: | ---: | ---: |
| student-mat | mid | decision_tree | 0.6084 | 0.6000 | 0.5500 |
| student-mat | late | gradient_boosting | 0.9372 | 0.9333 | 0.9000 |
| student-por | mid | gradient_boosting | 0.7105 | 0.7551 | 0.4667 |
| student-por | late | random_forest | 0.8581 | 0.8776 | 0.6667 |
| student-combined | mid | random_forest | 0.6957 | 0.7389 | 0.3714 |
| student-combined | late | gradient_boosting | 0.8422 | 0.8471 | 0.7714 |

Regression baseline:

| Dataset | Scenario | Model tot nhat | RMSE | MAE | R2 |
| --- | --- | --- | ---: | ---: | ---: |
| student-mat | mid | random_forest_regressor | 2.2016 | 1.6516 | 0.7087 |
| student-mat | late | gradient_boosting_regressor | 1.3843 | 1.0717 | 0.8848 |
| student-por | mid | random_forest_regressor | 2.1128 | 1.3814 | 0.6879 |
| student-por | late | ridge | 1.4579 | 0.9339 | 0.8514 |
| student-combined | mid | gradient_boosting_regressor | 2.0756 | 1.3837 | 0.7060 |
| student-combined | late | gradient_boosting_regressor | 1.1505 | 0.8035 | 0.9097 |

## 6. File ket qua chinh

- `reports/tables/final_imbalance_deep_comparison.csv`
- `reports/tables/deep_vs_tabular_comparison.csv`
- `reports/tables/deep_classification_summary.csv`
- `reports/tables/baseline_classification_summary.csv`
- `reports/tables/baseline_regression_summary.csv`
- `reports/tables/xapi_baseline_summary.csv`
- `reports/tables/xapi_deep_summary.csv`
- `report_temp/xapi_experiment_report.md`

## 7. Ket luan tien do

Pipeline hien tai da chay end-to-end thanh cong va co the dung cho bao cao tien do. Ket qua cho thay mo hinh deep `clsv2` co loi the ro o cac scenario `mid`, dac biet `student-mat/mid` va `student-por/mid`, trong khi cac baseline tabular van rat manh o scenario `late` do co them thong tin G2. Voi xAPI, `cnn_bilstm_xapi` dat Macro-F1 = 0.7708, thap hon baseline random forest borderline SMOTE Macro-F1 = 0.8235 nhung van co recall lop yeu cao 0.8947.
