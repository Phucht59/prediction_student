# Paired comparisons V2.1.1

Source keys: late_stage / cnn_bilstm_v2_tuned / g2_rule; late_stage / cnn_bilstm_v2_tuned / logistic_g2; late_stage / cnn_bilstm_v2_tuned / hgb_g1_g2; late_stage / cnn_bilstm_v2_tuned / small_mlp; late_stage / cnn_bilstm_v2_tuned / bilstm_only; late_stage / cnn_bilstm_v2_tuned / cnn_bilstm_legacy_config_v2_refit; late_stage / cnn_bilstm_v2_tuned / cnn_only; late_stage / g2_rule / hgb_g1_g2; late_stage / g2_rule / small_mlp; early_warning / g1_rule / small_mlp; early_warning / g1_rule / hgb_g1; early_warning / g1_rule / logistic_g1; early_warning / g1_rule / ridge_regression

| scenario | model_a | model_b | mean_difference | wins | ties | losses |
|---|---|---|---|---|---|---|
| late_stage | cnn_bilstm_v2_tuned | g2_rule | -0.09935748347695336 | 0 | 0 | 5 |
| late_stage | cnn_bilstm_v2_tuned | logistic_g2 | -0.09935748347695336 | 0 | 0 | 5 |
| late_stage | cnn_bilstm_v2_tuned | hgb_g1_g2 | -0.09134544738341008 | 0 | 0 | 5 |
| late_stage | cnn_bilstm_v2_tuned | small_mlp | -0.0894732832142628 | 0 | 0 | 5 |
| late_stage | cnn_bilstm_v2_tuned | bilstm_only | -0.03807824306973344 | 1 | 0 | 4 |
| late_stage | cnn_bilstm_v2_tuned | cnn_bilstm_legacy_config_v2_refit | 0.016393464693167447 | 3 | 0 | 2 |
| late_stage | cnn_bilstm_v2_tuned | cnn_only | 0.0440432226633765 | 4 | 0 | 1 |
| late_stage | g2_rule | hgb_g1_g2 | 0.008012036093543284 | 3 | 1 | 1 |
| late_stage | g2_rule | small_mlp | 0.009884200262690568 | 4 | 1 | 0 |
| early_warning | g1_rule | small_mlp | 0.0013506493506493245 | 1 | 4 | 0 |
| early_warning | g1_rule | hgb_g1 | 0.019739853346410728 | 1 | 4 | 0 |
| early_warning | g1_rule | logistic_g1 | 0.013974384668160322 | 3 | 2 | 0 |
| early_warning | g1_rule | ridge_regression | 0.0 | 0 | 5 | 0 |

