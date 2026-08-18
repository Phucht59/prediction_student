# OULAD Hybrid vs ML Stage Matrix

CNN-BiLSTM is compared with each tabular/deep comparator using 5,000 paired grouped bootstrap replicates over `id_student`. A confidence interval crossing zero is reported as insufficient evidence of difference; it is never described as equivalence.

| Stage | Comparator | Metric | Delta (CNN-BiLSTM − comparator) | 95% CI | Conclusion |
|---|---|---|---:|---|---|
| E1_EARLY_20PCT | bilstm_only_oulad | brier | 0.0017 | [0.0013, 0.0021] | comparator higher |
| E1_EARLY_20PCT | bilstm_only_oulad | macro_f1 | -0.0033 | [-0.0061, -0.0005] | comparator higher |
| E1_EARLY_20PCT | bilstm_only_oulad | nll | 0.0044 | [0.0034, 0.0053] | comparator higher |
| E1_EARLY_20PCT | bilstm_only_oulad | risk_f1 | -0.0074 | [-0.0111, -0.0038] | comparator higher |
| E1_EARLY_20PCT | bilstm_only_oulad | risk_recall | -0.0146 | [-0.0189, -0.0103] | comparator higher |
| E1_EARLY_20PCT | cnn_only_oulad | brier | 0.0034 | [0.0030, 0.0038] | comparator higher |
| E1_EARLY_20PCT | cnn_only_oulad | macro_f1 | 0.0010 | [-0.0018, 0.0038] | insufficient evidence of difference |
| E1_EARLY_20PCT | cnn_only_oulad | nll | 0.0075 | [0.0065, 0.0086] | comparator higher |
| E1_EARLY_20PCT | cnn_only_oulad | risk_f1 | 0.0036 | [-0.0001, 0.0072] | insufficient evidence of difference |
| E1_EARLY_20PCT | cnn_only_oulad | risk_recall | 0.0086 | [0.0044, 0.0128] | CNN-BiLSTM higher |
| E1_EARLY_20PCT | decision_tree_oulad | brier | -0.0248 | [-0.0285, -0.0213] | CNN-BiLSTM higher |
| E1_EARLY_20PCT | decision_tree_oulad | macro_f1 | 0.0893 | [0.0816, 0.0968] | CNN-BiLSTM higher |
| E1_EARLY_20PCT | decision_tree_oulad | nll | -0.3752 | [-0.4100, -0.3407] | CNN-BiLSTM higher |
| E1_EARLY_20PCT | decision_tree_oulad | risk_f1 | 0.1648 | [0.1538, 0.1758] | CNN-BiLSTM higher |
| E1_EARLY_20PCT | decision_tree_oulad | risk_recall | 0.2184 | [0.2074, 0.2294] | CNN-BiLSTM higher |
| E1_EARLY_20PCT | hist_gradient_boosting_oulad | brier | 0.0218 | [0.0194, 0.0241] | comparator higher |
| E1_EARLY_20PCT | hist_gradient_boosting_oulad | macro_f1 | -0.0014 | [-0.0062, 0.0035] | insufficient evidence of difference |
| E1_EARLY_20PCT | hist_gradient_boosting_oulad | nll | 0.0505 | [0.0450, 0.0561] | comparator higher |
| E1_EARLY_20PCT | hist_gradient_boosting_oulad | risk_f1 | 0.0043 | [-0.0021, 0.0106] | insufficient evidence of difference |
| E1_EARLY_20PCT | hist_gradient_boosting_oulad | risk_recall | 0.0180 | [0.0108, 0.0252] | CNN-BiLSTM higher |
| E1_EARLY_20PCT | logistic_regression_oulad | brier | 0.0129 | [0.0115, 0.0144] | comparator higher |
| E1_EARLY_20PCT | logistic_regression_oulad | macro_f1 | 0.0019 | [-0.0027, 0.0065] | insufficient evidence of difference |
| E1_EARLY_20PCT | logistic_regression_oulad | nll | 0.0291 | [0.0256, 0.0326] | comparator higher |
| E1_EARLY_20PCT | logistic_regression_oulad | risk_f1 | 0.0061 | [0.0001, 0.0122] | CNN-BiLSTM higher |
| E1_EARLY_20PCT | logistic_regression_oulad | risk_recall | 0.0143 | [0.0072, 0.0212] | CNN-BiLSTM higher |
| E1_EARLY_20PCT | mlp_oulad | brier | 0.0202 | [0.0179, 0.0226] | comparator higher |
| E1_EARLY_20PCT | mlp_oulad | macro_f1 | 0.0018 | [-0.0029, 0.0066] | insufficient evidence of difference |
| E1_EARLY_20PCT | mlp_oulad | nll | 0.0455 | [0.0397, 0.0512] | comparator higher |
| E1_EARLY_20PCT | mlp_oulad | risk_f1 | 0.0125 | [0.0062, 0.0189] | CNN-BiLSTM higher |
| E1_EARLY_20PCT | mlp_oulad | risk_recall | 0.0350 | [0.0278, 0.0424] | CNN-BiLSTM higher |
| E1_EARLY_20PCT | random_forest_oulad | brier | 0.0168 | [0.0147, 0.0188] | comparator higher |
| E1_EARLY_20PCT | random_forest_oulad | macro_f1 | 0.0059 | [0.0009, 0.0107] | CNN-BiLSTM higher |
| E1_EARLY_20PCT | random_forest_oulad | nll | 0.0382 | [0.0335, 0.0429] | comparator higher |
| E1_EARLY_20PCT | random_forest_oulad | risk_f1 | 0.0131 | [0.0066, 0.0195] | CNN-BiLSTM higher |
| E1_EARLY_20PCT | random_forest_oulad | risk_recall | 0.0249 | [0.0175, 0.0324] | CNN-BiLSTM higher |
| E1_EARLY_20PCT | svm_oulad | brier | 0.0144 | [0.0120, 0.0168] | comparator higher |
| E1_EARLY_20PCT | svm_oulad | macro_f1 | -0.0015 | [-0.0057, 0.0028] | insufficient evidence of difference |
| E1_EARLY_20PCT | svm_oulad | nll | 0.0289 | [0.0232, 0.0346] | comparator higher |
| E1_EARLY_20PCT | svm_oulad | risk_f1 | -0.0008 | [-0.0064, 0.0047] | insufficient evidence of difference |
| E1_EARLY_20PCT | svm_oulad | risk_recall | 0.0017 | [-0.0048, 0.0080] | insufficient evidence of difference |
| E1_EARLY_20PCT | xgboost_oulad | brier | 0.0225 | [0.0202, 0.0247] | comparator higher |
| E1_EARLY_20PCT | xgboost_oulad | macro_f1 | -0.0055 | [-0.0099, -0.0009] | comparator higher |
| E1_EARLY_20PCT | xgboost_oulad | nll | 0.0528 | [0.0475, 0.0581] | comparator higher |
| E1_EARLY_20PCT | xgboost_oulad | risk_f1 | -0.0048 | [-0.0106, 0.0011] | insufficient evidence of difference |
| E1_EARLY_20PCT | xgboost_oulad | risk_recall | 0.0007 | [-0.0060, 0.0075] | insufficient evidence of difference |
| E2_EARLY_35PCT | bilstm_only_oulad | brier | 0.0018 | [0.0015, 0.0022] | comparator higher |
| E2_EARLY_35PCT | bilstm_only_oulad | macro_f1 | 0.0001 | [-0.0033, 0.0034] | insufficient evidence of difference |
| E2_EARLY_35PCT | bilstm_only_oulad | nll | 0.0054 | [0.0045, 0.0063] | comparator higher |
| E2_EARLY_35PCT | bilstm_only_oulad | risk_f1 | -0.0096 | [-0.0138, -0.0056] | comparator higher |
| E2_EARLY_35PCT | bilstm_only_oulad | risk_recall | -0.0404 | [-0.0453, -0.0357] | comparator higher |
| E2_EARLY_35PCT | cnn_only_oulad | brier | 0.0008 | [0.0004, 0.0011] | comparator higher |
| E2_EARLY_35PCT | cnn_only_oulad | macro_f1 | -0.0014 | [-0.0038, 0.0010] | insufficient evidence of difference |
| E2_EARLY_35PCT | cnn_only_oulad | nll | 0.0020 | [0.0011, 0.0029] | comparator higher |
| E2_EARLY_35PCT | cnn_only_oulad | risk_f1 | 0.0005 | [-0.0025, 0.0036] | insufficient evidence of difference |
| E2_EARLY_35PCT | cnn_only_oulad | risk_recall | 0.0078 | [0.0042, 0.0113] | CNN-BiLSTM higher |
| E2_EARLY_35PCT | decision_tree_oulad | brier | -0.0347 | [-0.0383, -0.0311] | CNN-BiLSTM higher |
| E2_EARLY_35PCT | decision_tree_oulad | macro_f1 | 0.0561 | [0.0487, 0.0634] | CNN-BiLSTM higher |
| E2_EARLY_35PCT | decision_tree_oulad | nll | -0.4695 | [-0.5101, -0.4318] | CNN-BiLSTM higher |
| E2_EARLY_35PCT | decision_tree_oulad | risk_f1 | 0.0958 | [0.0862, 0.1058] | CNN-BiLSTM higher |
| E2_EARLY_35PCT | decision_tree_oulad | risk_recall | 0.1489 | [0.1380, 0.1598] | CNN-BiLSTM higher |
| E2_EARLY_35PCT | hist_gradient_boosting_oulad | brier | 0.0140 | [0.0121, 0.0159] | comparator higher |
| E2_EARLY_35PCT | hist_gradient_boosting_oulad | macro_f1 | -0.0078 | [-0.0124, -0.0031] | comparator higher |
| E2_EARLY_35PCT | hist_gradient_boosting_oulad | nll | 0.0350 | [0.0304, 0.0397] | comparator higher |
| E2_EARLY_35PCT | hist_gradient_boosting_oulad | risk_f1 | -0.0073 | [-0.0131, -0.0014] | comparator higher |
| E2_EARLY_35PCT | hist_gradient_boosting_oulad | risk_recall | 0.0011 | [-0.0059, 0.0083] | insufficient evidence of difference |
| E2_EARLY_35PCT | logistic_regression_oulad | brier | 0.0018 | [0.0007, 0.0028] | comparator higher |
| E2_EARLY_35PCT | logistic_regression_oulad | macro_f1 | -0.0000 | [-0.0041, 0.0042] | insufficient evidence of difference |
| E2_EARLY_35PCT | logistic_regression_oulad | nll | 0.0027 | [0.0002, 0.0053] | comparator higher |
| E2_EARLY_35PCT | logistic_regression_oulad | risk_f1 | -0.0004 | [-0.0056, 0.0050] | insufficient evidence of difference |
| E2_EARLY_35PCT | logistic_regression_oulad | risk_recall | -0.0015 | [-0.0080, 0.0051] | insufficient evidence of difference |
| E2_EARLY_35PCT | mlp_oulad | brier | 0.0136 | [0.0116, 0.0155] | comparator higher |
| E2_EARLY_35PCT | mlp_oulad | macro_f1 | -0.0043 | [-0.0089, 0.0002] | insufficient evidence of difference |
| E2_EARLY_35PCT | mlp_oulad | nll | 0.0332 | [0.0282, 0.0382] | comparator higher |
| E2_EARLY_35PCT | mlp_oulad | risk_f1 | 0.0003 | [-0.0055, 0.0060] | insufficient evidence of difference |
| E2_EARLY_35PCT | mlp_oulad | risk_recall | 0.0186 | [0.0119, 0.0254] | CNN-BiLSTM higher |
| E2_EARLY_35PCT | random_forest_oulad | brier | 0.0094 | [0.0078, 0.0110] | comparator higher |
| E2_EARLY_35PCT | random_forest_oulad | macro_f1 | 0.0024 | [-0.0023, 0.0071] | insufficient evidence of difference |
| E2_EARLY_35PCT | random_forest_oulad | nll | 0.0233 | [0.0193, 0.0272] | comparator higher |
| E2_EARLY_35PCT | random_forest_oulad | risk_f1 | 0.0047 | [-0.0014, 0.0107] | insufficient evidence of difference |
| E2_EARLY_35PCT | random_forest_oulad | risk_recall | 0.0095 | [0.0022, 0.0168] | CNN-BiLSTM higher |
| E2_EARLY_35PCT | svm_oulad | brier | 0.0094 | [0.0076, 0.0112] | comparator higher |
| E2_EARLY_35PCT | svm_oulad | macro_f1 | -0.0032 | [-0.0073, 0.0008] | insufficient evidence of difference |
| E2_EARLY_35PCT | svm_oulad | nll | 0.0192 | [0.0144, 0.0241] | comparator higher |
| E2_EARLY_35PCT | svm_oulad | risk_f1 | -0.0034 | [-0.0085, 0.0017] | insufficient evidence of difference |
| E2_EARLY_35PCT | svm_oulad | risk_recall | -0.0012 | [-0.0074, 0.0050] | insufficient evidence of difference |
| E2_EARLY_35PCT | xgboost_oulad | brier | 0.0146 | [0.0128, 0.0164] | comparator higher |
| E2_EARLY_35PCT | xgboost_oulad | macro_f1 | -0.0080 | [-0.0125, -0.0036] | comparator higher |
| E2_EARLY_35PCT | xgboost_oulad | nll | 0.0370 | [0.0327, 0.0414] | comparator higher |
| E2_EARLY_35PCT | xgboost_oulad | risk_f1 | -0.0121 | [-0.0176, -0.0065] | comparator higher |
| E2_EARLY_35PCT | xgboost_oulad | risk_recall | -0.0186 | [-0.0254, -0.0121] | comparator higher |
| L1_LATE_75PCT | bilstm_only_oulad | brier | 0.0010 | [0.0007, 0.0013] | comparator higher |
| L1_LATE_75PCT | bilstm_only_oulad | macro_f1 | 0.0115 | [0.0087, 0.0143] | CNN-BiLSTM higher |
| L1_LATE_75PCT | bilstm_only_oulad | nll | 0.0036 | [0.0028, 0.0044] | comparator higher |
| L1_LATE_75PCT | bilstm_only_oulad | risk_f1 | 0.0089 | [0.0058, 0.0119] | CNN-BiLSTM higher |
| L1_LATE_75PCT | bilstm_only_oulad | risk_recall | -0.0144 | [-0.0182, -0.0108] | comparator higher |
| L1_LATE_75PCT | cnn_only_oulad | brier | -0.0006 | [-0.0009, -0.0003] | CNN-BiLSTM higher |
| L1_LATE_75PCT | cnn_only_oulad | macro_f1 | 0.0038 | [0.0014, 0.0063] | CNN-BiLSTM higher |
| L1_LATE_75PCT | cnn_only_oulad | nll | -0.0011 | [-0.0019, -0.0003] | CNN-BiLSTM higher |
| L1_LATE_75PCT | cnn_only_oulad | risk_f1 | 0.0031 | [0.0005, 0.0058] | CNN-BiLSTM higher |
| L1_LATE_75PCT | cnn_only_oulad | risk_recall | -0.0040 | [-0.0073, -0.0009] | comparator higher |
| L1_LATE_75PCT | decision_tree_oulad | brier | -0.0280 | [-0.0309, -0.0252] | CNN-BiLSTM higher |
| L1_LATE_75PCT | decision_tree_oulad | macro_f1 | 0.0024 | [-0.0045, 0.0093] | insufficient evidence of difference |
| L1_LATE_75PCT | decision_tree_oulad | nll | -0.2753 | [-0.3063, -0.2451] | CNN-BiLSTM higher |
| L1_LATE_75PCT | decision_tree_oulad | risk_f1 | 0.0165 | [0.0083, 0.0245] | CNN-BiLSTM higher |
| L1_LATE_75PCT | decision_tree_oulad | risk_recall | 0.0896 | [0.0803, 0.0992] | CNN-BiLSTM higher |
| L1_LATE_75PCT | hist_gradient_boosting_oulad | brier | 0.0043 | [0.0031, 0.0055] | comparator higher |
| L1_LATE_75PCT | hist_gradient_boosting_oulad | macro_f1 | -0.0222 | [-0.0268, -0.0175] | comparator higher |
| L1_LATE_75PCT | hist_gradient_boosting_oulad | nll | 0.0144 | [0.0111, 0.0178] | comparator higher |
| L1_LATE_75PCT | hist_gradient_boosting_oulad | risk_f1 | -0.0191 | [-0.0244, -0.0138] | comparator higher |
| L1_LATE_75PCT | hist_gradient_boosting_oulad | risk_recall | 0.0240 | [0.0178, 0.0303] | CNN-BiLSTM higher |
| L1_LATE_75PCT | logistic_regression_oulad | brier | -0.0018 | [-0.0027, -0.0009] | CNN-BiLSTM higher |
| L1_LATE_75PCT | logistic_regression_oulad | macro_f1 | -0.0195 | [-0.0239, -0.0151] | comparator higher |
| L1_LATE_75PCT | logistic_regression_oulad | nll | -0.0028 | [-0.0057, 0.0001] | insufficient evidence of difference |
| L1_LATE_75PCT | logistic_regression_oulad | risk_f1 | -0.0149 | [-0.0200, -0.0100] | comparator higher |
| L1_LATE_75PCT | logistic_regression_oulad | risk_recall | 0.0339 | [0.0280, 0.0398] | CNN-BiLSTM higher |
| L1_LATE_75PCT | mlp_oulad | brier | 0.0053 | [0.0042, 0.0064] | comparator higher |
| L1_LATE_75PCT | mlp_oulad | macro_f1 | -0.0205 | [-0.0250, -0.0160] | comparator higher |
| L1_LATE_75PCT | mlp_oulad | nll | 0.0172 | [0.0136, 0.0208] | comparator higher |
| L1_LATE_75PCT | mlp_oulad | risk_f1 | -0.0190 | [-0.0243, -0.0138] | comparator higher |
| L1_LATE_75PCT | mlp_oulad | risk_recall | 0.0117 | [0.0059, 0.0176] | CNN-BiLSTM higher |
| L1_LATE_75PCT | random_forest_oulad | brier | 0.0022 | [0.0012, 0.0032] | comparator higher |
| L1_LATE_75PCT | random_forest_oulad | macro_f1 | -0.0237 | [-0.0284, -0.0191] | comparator higher |
| L1_LATE_75PCT | random_forest_oulad | nll | 0.0079 | [0.0049, 0.0109] | comparator higher |
| L1_LATE_75PCT | random_forest_oulad | risk_f1 | -0.0198 | [-0.0252, -0.0145] | comparator higher |
| L1_LATE_75PCT | random_forest_oulad | risk_recall | 0.0315 | [0.0255, 0.0377] | CNN-BiLSTM higher |
| L1_LATE_75PCT | svm_oulad | brier | 0.0002 | [-0.0009, 0.0013] | insufficient evidence of difference |
| L1_LATE_75PCT | svm_oulad | macro_f1 | -0.0256 | [-0.0302, -0.0209] | comparator higher |
| L1_LATE_75PCT | svm_oulad | nll | -0.0130 | [-0.0163, -0.0095] | CNN-BiLSTM higher |
| L1_LATE_75PCT | svm_oulad | risk_f1 | -0.0226 | [-0.0279, -0.0172] | comparator higher |
| L1_LATE_75PCT | svm_oulad | risk_recall | 0.0251 | [0.0190, 0.0314] | CNN-BiLSTM higher |
| L1_LATE_75PCT | xgboost_oulad | brier | 0.0049 | [0.0038, 0.0061] | comparator higher |
| L1_LATE_75PCT | xgboost_oulad | macro_f1 | -0.0243 | [-0.0286, -0.0199] | comparator higher |
| L1_LATE_75PCT | xgboost_oulad | nll | 0.0166 | [0.0136, 0.0197] | comparator higher |
| L1_LATE_75PCT | xgboost_oulad | risk_f1 | -0.0216 | [-0.0264, -0.0167] | comparator higher |
| L1_LATE_75PCT | xgboost_oulad | risk_recall | 0.0223 | [0.0167, 0.0279] | CNN-BiLSTM higher |
| M1_MIDDLE_FROZEN | bilstm_only_oulad | brier | 0.0013 | [0.0010, 0.0017] | comparator higher |
| M1_MIDDLE_FROZEN | bilstm_only_oulad | macro_f1 | 0.0082 | [0.0052, 0.0113] | CNN-BiLSTM higher |
| M1_MIDDLE_FROZEN | bilstm_only_oulad | nll | 0.0042 | [0.0034, 0.0051] | comparator higher |
| M1_MIDDLE_FROZEN | bilstm_only_oulad | risk_f1 | 0.0019 | [-0.0016, 0.0054] | insufficient evidence of difference |
| M1_MIDDLE_FROZEN | bilstm_only_oulad | risk_recall | -0.0309 | [-0.0355, -0.0265] | comparator higher |
| M1_MIDDLE_FROZEN | cnn_only_oulad | brier | 0.0003 | [-0.0000, 0.0006] | insufficient evidence of difference |
| M1_MIDDLE_FROZEN | cnn_only_oulad | macro_f1 | -0.0025 | [-0.0049, -0.0001] | comparator higher |
| M1_MIDDLE_FROZEN | cnn_only_oulad | nll | 0.0008 | [-0.0001, 0.0016] | insufficient evidence of difference |
| M1_MIDDLE_FROZEN | cnn_only_oulad | risk_f1 | -0.0020 | [-0.0048, 0.0008] | insufficient evidence of difference |
| M1_MIDDLE_FROZEN | cnn_only_oulad | risk_recall | 0.0026 | [-0.0008, 0.0061] | insufficient evidence of difference |
| M1_MIDDLE_FROZEN | decision_tree_oulad | brier | -0.0385 | [-0.0417, -0.0352] | CNN-BiLSTM higher |
| M1_MIDDLE_FROZEN | decision_tree_oulad | macro_f1 | 0.0323 | [0.0254, 0.0391] | CNN-BiLSTM higher |
| M1_MIDDLE_FROZEN | decision_tree_oulad | nll | -0.4231 | [-0.4608, -0.3878] | CNN-BiLSTM higher |
| M1_MIDDLE_FROZEN | decision_tree_oulad | risk_f1 | 0.0636 | [0.0550, 0.0723] | CNN-BiLSTM higher |
| M1_MIDDLE_FROZEN | decision_tree_oulad | risk_recall | 0.1529 | [0.1425, 0.1633] | CNN-BiLSTM higher |
| M1_MIDDLE_FROZEN | hist_gradient_boosting_oulad | brier | 0.0057 | [0.0042, 0.0071] | comparator higher |
| M1_MIDDLE_FROZEN | hist_gradient_boosting_oulad | macro_f1 | -0.0072 | [-0.0116, -0.0027] | comparator higher |
| M1_MIDDLE_FROZEN | hist_gradient_boosting_oulad | nll | 0.0165 | [0.0126, 0.0204] | comparator higher |
| M1_MIDDLE_FROZEN | hist_gradient_boosting_oulad | risk_f1 | -0.0013 | [-0.0066, 0.0040] | insufficient evidence of difference |
| M1_MIDDLE_FROZEN | hist_gradient_boosting_oulad | risk_recall | 0.0315 | [0.0247, 0.0385] | CNN-BiLSTM higher |
| M1_MIDDLE_FROZEN | logistic_regression_oulad | brier | -0.0018 | [-0.0027, -0.0009] | CNN-BiLSTM higher |
| M1_MIDDLE_FROZEN | logistic_regression_oulad | macro_f1 | -0.0024 | [-0.0065, 0.0016] | insufficient evidence of difference |
| M1_MIDDLE_FROZEN | logistic_regression_oulad | nll | -0.0052 | [-0.0077, -0.0027] | CNN-BiLSTM higher |
| M1_MIDDLE_FROZEN | logistic_regression_oulad | risk_f1 | 0.0030 | [-0.0020, 0.0077] | insufficient evidence of difference |
| M1_MIDDLE_FROZEN | logistic_regression_oulad | risk_recall | 0.0286 | [0.0224, 0.0347] | CNN-BiLSTM higher |
| M1_MIDDLE_FROZEN | mlp_oulad | brier | 0.0061 | [0.0047, 0.0074] | comparator higher |
| M1_MIDDLE_FROZEN | mlp_oulad | macro_f1 | -0.0067 | [-0.0110, -0.0023] | comparator higher |
| M1_MIDDLE_FROZEN | mlp_oulad | nll | 0.0170 | [0.0132, 0.0207] | comparator higher |
| M1_MIDDLE_FROZEN | mlp_oulad | risk_f1 | -0.0012 | [-0.0063, 0.0041] | insufficient evidence of difference |
| M1_MIDDLE_FROZEN | mlp_oulad | risk_recall | 0.0293 | [0.0228, 0.0360] | CNN-BiLSTM higher |
| M1_MIDDLE_FROZEN | random_forest_oulad | brier | 0.0033 | [0.0019, 0.0046] | comparator higher |
| M1_MIDDLE_FROZEN | random_forest_oulad | macro_f1 | -0.0036 | [-0.0084, 0.0011] | insufficient evidence of difference |
| M1_MIDDLE_FROZEN | random_forest_oulad | nll | 0.0075 | [0.0031, 0.0117] | comparator higher |
| M1_MIDDLE_FROZEN | random_forest_oulad | risk_f1 | 0.0019 | [-0.0037, 0.0076] | insufficient evidence of difference |
| M1_MIDDLE_FROZEN | random_forest_oulad | risk_recall | 0.0296 | [0.0224, 0.0369] | CNN-BiLSTM higher |
| M1_MIDDLE_FROZEN | svm_oulad | brier | 0.0030 | [0.0017, 0.0044] | comparator higher |
| M1_MIDDLE_FROZEN | svm_oulad | macro_f1 | -0.0061 | [-0.0098, -0.0020] | comparator higher |
| M1_MIDDLE_FROZEN | svm_oulad | nll | 0.0012 | [-0.0025, 0.0051] | insufficient evidence of difference |
| M1_MIDDLE_FROZEN | svm_oulad | risk_f1 | -0.0024 | [-0.0068, 0.0024] | insufficient evidence of difference |
| M1_MIDDLE_FROZEN | svm_oulad | risk_recall | 0.0194 | [0.0137, 0.0254] | CNN-BiLSTM higher |
| M1_MIDDLE_FROZEN | xgboost_oulad | brier | 0.0064 | [0.0049, 0.0078] | comparator higher |
| M1_MIDDLE_FROZEN | xgboost_oulad | macro_f1 | -0.0046 | [-0.0089, -0.0004] | comparator higher |
| M1_MIDDLE_FROZEN | xgboost_oulad | nll | 0.0184 | [0.0148, 0.0222] | comparator higher |
| M1_MIDDLE_FROZEN | xgboost_oulad | risk_f1 | -0.0005 | [-0.0056, 0.0046] | insufficient evidence of difference |
| M1_MIDDLE_FROZEN | xgboost_oulad | risk_recall | 0.0222 | [0.0157, 0.0288] | CNN-BiLSTM higher |
