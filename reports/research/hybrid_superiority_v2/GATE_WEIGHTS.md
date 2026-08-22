# Gate weights by cutoff

Nguồn chính: `test_lab/artifacts/hybrid_vnext/phase4/GATE_DIAGNOSTICS.csv` (L1_control, 9 run).
Hình: `C:/hufit/student/reports/research/hybrid_superiority_v2/figures/gate_weights_by_cutoff.png`.

| dataset | stage | tabular | CNN | BiLSTM |
|---|---|---:|---:|---:|
| oulad | 100pct | 0.172 | 0.237 | 0.591 |
| oulad | 20pct | 0.315 | 0.232 | 0.453 |
| oulad | 35pct | 0.272 | 0.245 | 0.483 |
| oulad | 50pct | 0.232 | 0.251 | 0.517 |
| oulad | 75pct | 0.200 | 0.251 | 0.549 |
| uci | S0 | 1.000 | 0.000 | 0.000 |
| uci | S1 | 0.064 | 0.263 | 0.673 |
| uci | S2 | 0.057 | 0.250 | 0.693 |

UCI S0: tabular_mass = 1 (CNN/BiLSTM tắt, T=0) — đúng thiết kế.
Nếu CNN+BiLSTM mass tăng từ 20%→75% trên OULAD, cổng đang chuyển sang chuỗi khi có tuần VLE.

Serving checkpoints: c0_inner_fold0_seed42.pt keys=['model_id', 'instance', 'fold', 'seed', 'state_dict', 'config'] model_id=hybrid; c0_inner_fold1_seed42.pt keys=['model_id', 'instance', 'fold', 'seed', 'state_dict', 'config'] model_id=hybrid; c0_inner_fold2_seed42.pt keys=['model_id', 'instance', 'fold', 'seed', 'state_dict', 'config'] model_id=hybrid
