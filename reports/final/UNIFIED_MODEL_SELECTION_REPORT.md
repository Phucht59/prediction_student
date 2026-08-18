# Unified Model Selection Report

The selection objective is the mean Macro-F1 across S0/S1/S2 on three inner folds. A single configuration and estimator serve all stages.

| Dataset | Model | Outer fold | Inner mean stage Macro-F1 |
|---|---|---:|---:|
| student_mat | logistic_regression_mat | 0 | 0.6658 |
| student_mat | logistic_regression_mat | 1 | 0.6923 |
| student_mat | logistic_regression_mat | 2 | 0.6645 |
| student_mat | logistic_regression_mat | 3 | 0.6716 |
| student_mat | logistic_regression_mat | 4 | 0.6827 |
| student_mat | decision_tree_mat | 0 | 0.6563 |
| student_mat | decision_tree_mat | 1 | 0.6469 |
| student_mat | decision_tree_mat | 2 | 0.6401 |
| student_mat | decision_tree_mat | 3 | 0.6435 |
| student_mat | decision_tree_mat | 4 | 0.6501 |
| student_mat | random_forest_mat | 0 | 0.6854 |
| student_mat | random_forest_mat | 1 | 0.7062 |
| student_mat | random_forest_mat | 2 | 0.6747 |
| student_mat | random_forest_mat | 3 | 0.6991 |
| student_mat | random_forest_mat | 4 | 0.6758 |
| student_mat | hist_gradient_boosting_mat | 0 | 0.6787 |
| student_mat | hist_gradient_boosting_mat | 1 | 0.6564 |
| student_mat | hist_gradient_boosting_mat | 2 | 0.6365 |
| student_mat | hist_gradient_boosting_mat | 3 | 0.6598 |
| student_mat | hist_gradient_boosting_mat | 4 | 0.6200 |
| student_mat | svm_mat | 0 | 0.6987 |
| student_mat | svm_mat | 1 | 0.6807 |
| student_mat | svm_mat | 2 | 0.6512 |
| student_mat | svm_mat | 3 | 0.6586 |
| student_mat | svm_mat | 4 | 0.6445 |
| student_mat | xgboost_mat | 0 | 0.6935 |
| student_mat | xgboost_mat | 1 | 0.7023 |
| student_mat | xgboost_mat | 2 | 0.6574 |
| student_mat | xgboost_mat | 3 | 0.6851 |
| student_mat | xgboost_mat | 4 | 0.6411 |
| student_mat | mlp_mat | 0 | 0.6880 |
| student_mat | mlp_mat | 1 | 0.6960 |
| student_mat | mlp_mat | 2 | 0.6679 |
| student_mat | mlp_mat | 3 | 0.6588 |
| student_mat | mlp_mat | 4 | 0.6818 |
| student_mat | cnn_only_mat | 0 | 0.6803 |
| student_mat | cnn_only_mat | 1 | 0.6940 |
| student_mat | cnn_only_mat | 2 | 0.6642 |
| student_mat | cnn_only_mat | 3 | 0.6706 |
| student_mat | cnn_only_mat | 4 | 0.6526 |
| student_mat | bilstm_only_mat | 0 | 0.5391 |
| student_mat | bilstm_only_mat | 1 | 0.6461 |
| student_mat | bilstm_only_mat | 2 | 0.5582 |
| student_mat | bilstm_only_mat | 3 | 0.5997 |
| student_mat | bilstm_only_mat | 4 | 0.5887 |
| student_mat | cnn_bilstm_mat | 0 | 0.6456 |
| student_mat | cnn_bilstm_mat | 1 | 0.6501 |
| student_mat | cnn_bilstm_mat | 2 | 0.6400 |
| student_mat | cnn_bilstm_mat | 3 | 0.6568 |
| student_mat | cnn_bilstm_mat | 4 | 0.6716 |
| student_por | logistic_regression_por | 0 | 0.6499 |
| student_por | logistic_regression_por | 1 | 0.6669 |
| student_por | logistic_regression_por | 2 | 0.6747 |
| student_por | logistic_regression_por | 3 | 0.6512 |
| student_por | logistic_regression_por | 4 | 0.6799 |
| student_por | decision_tree_por | 0 | 0.6457 |
| student_por | decision_tree_por | 1 | 0.6549 |
| student_por | decision_tree_por | 2 | 0.6267 |
| student_por | decision_tree_por | 3 | 0.6501 |
| student_por | decision_tree_por | 4 | 0.6579 |
| student_por | random_forest_por | 0 | 0.6987 |
| student_por | random_forest_por | 1 | 0.7008 |
| student_por | random_forest_por | 2 | 0.7134 |
| student_por | random_forest_por | 3 | 0.7087 |
| student_por | random_forest_por | 4 | 0.7256 |
| student_por | hist_gradient_boosting_por | 0 | 0.6481 |
| student_por | hist_gradient_boosting_por | 1 | 0.6537 |
| student_por | hist_gradient_boosting_por | 2 | 0.6429 |
| student_por | hist_gradient_boosting_por | 3 | 0.6352 |
| student_por | hist_gradient_boosting_por | 4 | 0.6620 |
| student_por | svm_por | 0 | 0.6515 |
| student_por | svm_por | 1 | 0.6553 |
| student_por | svm_por | 2 | 0.6474 |
| student_por | svm_por | 3 | 0.6533 |
| student_por | svm_por | 4 | 0.6784 |
| student_por | xgboost_por | 0 | 0.6781 |
| student_por | xgboost_por | 1 | 0.6714 |
| student_por | xgboost_por | 2 | 0.6678 |
| student_por | xgboost_por | 3 | 0.6759 |
| student_por | xgboost_por | 4 | 0.6916 |
| student_por | mlp_por | 0 | 0.6585 |
| student_por | mlp_por | 1 | 0.6563 |
| student_por | mlp_por | 2 | 0.6447 |
| student_por | mlp_por | 3 | 0.6648 |
| student_por | mlp_por | 4 | 0.6974 |
| student_por | cnn_only_por | 0 | 0.6691 |
| student_por | cnn_only_por | 1 | 0.6866 |
| student_por | cnn_only_por | 2 | 0.6761 |
| student_por | cnn_only_por | 3 | 0.7018 |
| student_por | cnn_only_por | 4 | 0.7078 |
| student_por | bilstm_only_por | 0 | 0.6636 |
| student_por | bilstm_only_por | 1 | 0.5732 |
| student_por | bilstm_only_por | 2 | 0.6699 |
| student_por | bilstm_only_por | 3 | 0.6700 |
| student_por | bilstm_only_por | 4 | 0.6935 |
| student_por | cnn_bilstm_por | 0 | 0.6766 |
| student_por | cnn_bilstm_por | 1 | 0.6838 |
| student_por | cnn_bilstm_por | 2 | 0.6875 |
| student_por | cnn_bilstm_por | 3 | 0.6877 |
| student_por | cnn_bilstm_por | 4 | 0.7051 |

No outer score, best seed, transfer checkpoint, pretrained checkpoint, synthetic oversampling, ordinal auxiliary head, regression auxiliary head, or grade-band prior was used for selection.

## Grade-band diagnostic reference

The following training-fold-only reference is reported without a model identity:

- student_mat S1_MID_G1_ONLY: Macro-F1 0.7666.
- student_mat S2_LATE_G1_G2: Macro-F1 0.9067.
- student_por S1_MID_G1_ONLY: Macro-F1 0.7400.
- student_por S2_LATE_G1_G2: Macro-F1 0.8166.
