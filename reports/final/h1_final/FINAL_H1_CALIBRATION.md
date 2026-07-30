# Final H1 Calibration

| Model | Macro-F1 | PR-AUC | ROC-AUC | NLL | Brier | ECE |
| --- | --- | --- | --- | --- | --- | --- |
| MLP | 0.777599 | 0.839308 | 0.852435 | 0.436951 | 0.142697 | 0.017776 |
| H0 Current Hybrid | 0.774879 | 0.837280 | 0.851260 | 0.439604 | 0.143928 | 0.025288 |
| H1 Tabular Residual Hybrid | 0.777138 | 0.838929 | 0.852624 | 0.438923 | 0.143691 | 0.029090 |

Against H0, H1 slightly improves NLL by
**0.000681** and Brier by
**0.000237**, while ECE worsens
by **0.003802**.

Against MLP, H1 is worse in NLL by
**0.001972**, Brier by
**0.000994**, and ECE by
**0.011314**. No post-hoc calibration method
was introduced in Phase 6.
