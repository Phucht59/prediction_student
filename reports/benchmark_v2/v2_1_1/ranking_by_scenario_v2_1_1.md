# Ranking by scenario V2.1.1

| scenario | model | feature_set_id | macro_f1_mean | macro_f1_sd_across_outer_folds |
|---|---|---|---|---|
| early_warning | g1_rule | G1 | 0.7567158013231956 | 0.04876539517873498 |
| early_warning | ridge_regression | G1 | 0.7567158013231956 | 0.04876539517873498 |
| early_warning | small_mlp | G1 | 0.7553651519725464 | 0.0491449029571362 |
| early_warning | logistic_g1 | G1 | 0.7427414166550353 | 0.04853341721627835 |
| early_warning | ordinal_logistic | G1 | 0.7427414166550353 | 0.04853341721627835 |
| early_warning | hgb_g1 | G1 | 0.7369759479767849 | 0.07591050968995239 |
| early_warning | majority | G1 | 0.21843441823501789 | 0.002139100824226131 |
| late_stage | g2_rule | G2 | 0.8977413178891289 | 0.02374898012983326 |
| late_stage | logistic_g2 | G2 | 0.8977413178891289 | 0.02374898012983326 |
| late_stage | hgb_g1_g2 | G1+G2 | 0.8897292817955854 | 0.028715967262010283 |
| late_stage | small_mlp | G1+G2 | 0.8878571176264384 | 0.02044493524558438 |
| late_stage | logistic_g1_g2 | G1+G2 | 0.8860952107860228 | 0.03182305985832529 |
| late_stage | ordinal_logistic | G1+G2 | 0.8860952107860228 | 0.03182305985832529 |
| late_stage | ridge_regression | G1+G2 | 0.8714746077998351 | 0.02721840006472166 |
| late_stage | bilstm_only | G1+G2 | 0.8364620774819089 | 0.013506894900162767 |
| late_stage | cnn_bilstm_v2_tuned | G1+G2 | 0.7983838344121756 | 0.05262412304718128 |
| late_stage | cnn_bilstm_legacy_config_v2_refit | G1+G2 | 0.781990369719008 | 0.04199583782257365 |
| late_stage | cnn_only | G1+G2 | 0.754340611748799 | 0.04010887763798975 |
| late_stage | majority | G1+G2 | 0.21843441823501789 | 0.002139100824226131 |
