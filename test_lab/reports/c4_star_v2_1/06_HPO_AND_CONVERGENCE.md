# 06 HPO and convergence

Optuna constrained J, fold 0 seed 42, inner VALID. Study names `c4_v21_hpo_{domain}_ce758268ce0c`.

UCI complete trials (last snapshot): **158**. Best J `-68.221`.
Params: `{"bilstm_hidden": 32, "cnn_channels": 16, "d_fuse": 48, "dropout": 0.1053726445918855, "initial_alpha": 0.05, "kd_temperature": 3.0, "lambda_aux": 0.12369737868719721, "lambda_kd": 1.0, "lambda_rank": 0.2, "lambda_ssl": 0.25, "lr": 0.00021988802646132354, "mechanism": "M4", "weight_decay": 2.119835098881838e-05}`
AP: `{"S0": 0.4536534197717586, "S1": 0.6972744178788622, "S2": 0.8041386902660019}`

UCI still **loses both warm stages** vs CatBoost (S1 0.769 / S2 0.907).

OULAD complete trials (last snapshot): **156**. Best J `0.307`.
Params: `{"bilstm_hidden": 48, "cnn_channels": 16, "d_fuse": 48, "dropout": 0.27214077811588067, "initial_alpha": 0.02, "kd_temperature": 2.0, "lambda_aux": 0.1051225642159071, "lambda_kd": 1.0, "lambda_rank": 0.1, "lambda_ssl": 0.1, "lr": 0.00027575876144757597, "mechanism": "M4", "weight_decay": 0.00041229127738522076}`
AP: `{"100pct": 0.926420276530032, "20pct": 0.7614592206506516, "35pct": 0.8098129278920283, "50pct": 0.8592832920219452, "75pct": 0.9006963069883003}`

OULAD HPO winner (fold 0 / seed 42) has **n_warm_loss=0** vs the v2.1 3×3 ceiling, but normalized margins r_s ≪ 1 (not Gold).
That single-fold result is **not** confirmation. See 07 for 3×3.
