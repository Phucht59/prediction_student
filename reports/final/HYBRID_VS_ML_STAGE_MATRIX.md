# Hybrid vs ML Stage Matrix

Paired intervals use 5,000 base-record bootstrap replicates. Overall resampling keeps all three views of each sampled base record together.

| Dataset | Stage | Comparator | Delta Macro-F1 | 95% CI | Conclusion |
|---|---|---|---:|---|---|
| student_mat | S0_EARLY_NO_GRADE | logistic_regression_mat | 0.0156 | [-0.0256, 0.0568] | insufficient evidence of difference |
| student_mat | S1_MID_G1_ONLY | logistic_regression_mat | 0.0214 | [-0.0077, 0.0512] | insufficient evidence of difference |
| student_mat | S2_LATE_G1_G2 | logistic_regression_mat | -0.0293 | [-0.0542, -0.0057] | CNN-BiLSTM lower |
| student_mat | S0_EARLY_NO_GRADE | decision_tree_mat | 0.0032 | [-0.0526, 0.0595] | insufficient evidence of difference |
| student_mat | S1_MID_G1_ONLY | decision_tree_mat | 0.0128 | [-0.0288, 0.0542] | insufficient evidence of difference |
| student_mat | S2_LATE_G1_G2 | decision_tree_mat | -0.0161 | [-0.0543, 0.0215] | insufficient evidence of difference |
| student_mat | S0_EARLY_NO_GRADE | random_forest_mat | -0.0161 | [-0.0609, 0.0302] | insufficient evidence of difference |
| student_mat | S1_MID_G1_ONLY | random_forest_mat | 0.0320 | [-0.0013, 0.0651] | insufficient evidence of difference |
| student_mat | S2_LATE_G1_G2 | random_forest_mat | -0.0432 | [-0.0728, -0.0139] | CNN-BiLSTM lower |
| student_mat | S0_EARLY_NO_GRADE | hist_gradient_boosting_mat | -0.0243 | [-0.0760, 0.0298] | insufficient evidence of difference |
| student_mat | S1_MID_G1_ONLY | hist_gradient_boosting_mat | 0.0424 | [-0.0042, 0.0905] | insufficient evidence of difference |
| student_mat | S2_LATE_G1_G2 | hist_gradient_boosting_mat | -0.0080 | [-0.0458, 0.0297] | insufficient evidence of difference |
| student_mat | S0_EARLY_NO_GRADE | svm_mat | -0.0387 | [-0.0870, 0.0111] | insufficient evidence of difference |
| student_mat | S1_MID_G1_ONLY | svm_mat | 0.0194 | [-0.0225, 0.0625] | insufficient evidence of difference |
| student_mat | S2_LATE_G1_G2 | svm_mat | -0.0040 | [-0.0404, 0.0336] | insufficient evidence of difference |
| student_mat | S0_EARLY_NO_GRADE | xgboost_mat | 0.0020 | [-0.0490, 0.0551] | insufficient evidence of difference |
| student_mat | S1_MID_G1_ONLY | xgboost_mat | 0.0346 | [-0.0095, 0.0783] | insufficient evidence of difference |
| student_mat | S2_LATE_G1_G2 | xgboost_mat | -0.0279 | [-0.0617, 0.0048] | insufficient evidence of difference |
| student_mat | S0_EARLY_NO_GRADE | mlp_mat | -0.0083 | [-0.0578, 0.0440] | insufficient evidence of difference |
| student_mat | S1_MID_G1_ONLY | mlp_mat | 0.0132 | [-0.0262, 0.0515] | insufficient evidence of difference |
| student_mat | S2_LATE_G1_G2 | mlp_mat | -0.0086 | [-0.0400, 0.0224] | insufficient evidence of difference |
| student_mat | S0_EARLY_NO_GRADE | cnn_only_mat | -0.0103 | [-0.0433, 0.0231] | insufficient evidence of difference |
| student_mat | S1_MID_G1_ONLY | cnn_only_mat | -0.0084 | [-0.0270, 0.0094] | insufficient evidence of difference |
| student_mat | S2_LATE_G1_G2 | cnn_only_mat | -0.0056 | [-0.0206, 0.0088] | insufficient evidence of difference |
| student_mat | S0_EARLY_NO_GRADE | bilstm_only_mat | -0.0177 | [-0.0503, 0.0157] | insufficient evidence of difference |
| student_mat | S1_MID_G1_ONLY | bilstm_only_mat | 0.0753 | [0.0442, 0.1088] | CNN-BiLSTM higher |
| student_mat | S2_LATE_G1_G2 | bilstm_only_mat | 0.0913 | [0.0606, 0.1237] | CNN-BiLSTM higher |
| student_por | S0_EARLY_NO_GRADE | logistic_regression_por | 0.0738 | [0.0412, 0.1060] | CNN-BiLSTM higher |
| student_por | S1_MID_G1_ONLY | logistic_regression_por | -0.0030 | [-0.0235, 0.0166] | insufficient evidence of difference |
| student_por | S2_LATE_G1_G2 | logistic_regression_por | 0.0115 | [-0.0077, 0.0304] | insufficient evidence of difference |
| student_por | S0_EARLY_NO_GRADE | decision_tree_por | 0.0777 | [0.0364, 0.1204] | CNN-BiLSTM higher |
| student_por | S1_MID_G1_ONLY | decision_tree_por | 0.0344 | [0.0014, 0.0662] | CNN-BiLSTM higher |
| student_por | S2_LATE_G1_G2 | decision_tree_por | 0.0565 | [0.0249, 0.0883] | CNN-BiLSTM higher |
| student_por | S0_EARLY_NO_GRADE | random_forest_por | 0.0007 | [-0.0241, 0.0251] | insufficient evidence of difference |
| student_por | S1_MID_G1_ONLY | random_forest_por | -0.0294 | [-0.0507, -0.0086] | CNN-BiLSTM lower |
| student_por | S2_LATE_G1_G2 | random_forest_por | -0.0051 | [-0.0232, 0.0143] | insufficient evidence of difference |
| student_por | S0_EARLY_NO_GRADE | hist_gradient_boosting_por | 0.0503 | [0.0029, 0.0986] | CNN-BiLSTM higher |
| student_por | S1_MID_G1_ONLY | hist_gradient_boosting_por | 0.0709 | [0.0313, 0.1111] | CNN-BiLSTM higher |
| student_por | S2_LATE_G1_G2 | hist_gradient_boosting_por | 0.0338 | [0.0017, 0.0680] | CNN-BiLSTM higher |
| student_por | S0_EARLY_NO_GRADE | svm_por | 0.0596 | [0.0125, 0.1075] | CNN-BiLSTM higher |
| student_por | S1_MID_G1_ONLY | svm_por | 0.0012 | [-0.0315, 0.0351] | insufficient evidence of difference |
| student_por | S2_LATE_G1_G2 | svm_por | 0.0521 | [0.0169, 0.0898] | CNN-BiLSTM higher |
| student_por | S0_EARLY_NO_GRADE | xgboost_por | 0.0555 | [0.0077, 0.1058] | CNN-BiLSTM higher |
| student_por | S1_MID_G1_ONLY | xgboost_por | -0.0106 | [-0.0436, 0.0231] | insufficient evidence of difference |
| student_por | S2_LATE_G1_G2 | xgboost_por | 0.0091 | [-0.0193, 0.0403] | insufficient evidence of difference |
| student_por | S0_EARLY_NO_GRADE | mlp_por | 0.1086 | [0.0592, 0.1588] | CNN-BiLSTM higher |
| student_por | S1_MID_G1_ONLY | mlp_por | 0.0018 | [-0.0329, 0.0386] | insufficient evidence of difference |
| student_por | S2_LATE_G1_G2 | mlp_por | -0.0001 | [-0.0265, 0.0276] | insufficient evidence of difference |
| student_por | S0_EARLY_NO_GRADE | cnn_only_por | 0.0105 | [-0.0106, 0.0324] | insufficient evidence of difference |
| student_por | S1_MID_G1_ONLY | cnn_only_por | -0.0123 | [-0.0246, -0.0002] | CNN-BiLSTM lower |
| student_por | S2_LATE_G1_G2 | cnn_only_por | 0.0176 | [0.0014, 0.0343] | CNN-BiLSTM higher |
| student_por | S0_EARLY_NO_GRADE | bilstm_only_por | -0.0005 | [-0.0204, 0.0190] | insufficient evidence of difference |
| student_por | S1_MID_G1_ONLY | bilstm_only_por | 0.0134 | [-0.0015, 0.0285] | insufficient evidence of difference |
| student_por | S2_LATE_G1_G2 | bilstm_only_por | 0.0328 | [0.0139, 0.0516] | CNN-BiLSTM higher |
