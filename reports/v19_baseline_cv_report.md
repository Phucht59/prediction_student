# V19 Baseline Cross-Validation Report

Protocol: stratified 5-fold cross-validation. The preprocessor is fit inside each fold, then the model is evaluated on that fold's holdout split.

| dataset | model | folds | accuracy_mean | accuracy_std | f1_macro_mean | f1_macro_std |
| --- | --- | --- | --- | --- | --- | --- |
| student-mat | DecisionTree | 5 | 0.7443 | 0.0314 | 0.7239 | 0.0249 |
| student-mat | RandomForest | 5 | 0.7620 | 0.0411 | 0.7475 | 0.0416 |
| student-por | DecisionTree | 5 | 0.7412 | 0.0293 | 0.7317 | 0.0401 |
| student-por | RandomForest | 5 | 0.7196 | 0.0379 | 0.7156 | 0.0361 |
| xapi | DecisionTree | 5 | 0.6229 | 0.0275 | 0.6319 | 0.0265 |
| xapi | RandomForest | 5 | 0.6687 | 0.0153 | 0.6788 | 0.0157 |

## Notes

- This satisfies the baseline CV requirement for Decision Tree and Random Forest.
- These are baseline ML results, not CNN-BiLSTM results.
- Feature set is the paper feature set to keep the comparison simple and reproducible.
