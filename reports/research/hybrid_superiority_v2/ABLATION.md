# Ablation

Independent retrain của các nhánh preregister. **SPEED_FINISH: UCI fold 0 / seed 42 / 8 epoch** — không phải ablation 3×3 confirmation.

Hparams lấy từ C0-R UCI 3×3 (`lr≈5.1e-5`). 8 epoch **không đủ** để `full` hội tụ; bảng này chỉ để provenance SPEED, không để claim.

| Ablation | branch_mode | S0 | S1 | S2 | Ghi chú |
|---|---|---:|---:|---:|---|
| tabular_only | tabular | 0.278 | 0.278 | 0.278 | không grade → ~prevalence |
| tabular_cnn | cnn | 0.280 | 0.323 | 0.330 | undertrain |
| tabular_bilstm | bilstm | 0.283 | 0.280 | 0.279 | undertrain |
| serial_no_tabular | temporal | 0.232 | **0.740** | **0.904** | G1/G2 nằm temporal — S2 gần CatBoost |
| full | full | 0.279 | 0.321 | 0.322 | 8 epoch, lr thấp; **không** so 3×3 0.913 |
| full_no_gate | full | 0.279 | 0.321 | 0.322 | SPEED không tách softmax-gate |
| full_no_rank | full | 0.282 | 0.324 | 0.324 | |
| full_no_kd | full | 0.279 | 0.321 | 0.322 | λ_kd=0 sẵn trong SPEED |
| full_no_multiprefix | full | 0.248 | 0.246 | 0.246 | |

Kết luận an toàn: tín hiệu UCI S2 chủ yếu từ **temporal grades**, không từ tabular. C0-R 3×3 (12+ epoch, 9 run) mới là số Hybrid UCI được gate. Ablation SPEED **không** pass ngưỡng 0.005 vs full.

OULAD ablation: không chạy trong SPEED_FINISH.
