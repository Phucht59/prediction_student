# Final H1 Evaluation

Phase 6 completed all **45/45** predefined model/fold/seed runs with zero
failures. No post-outer tuning occurred.

| Model | Macro-F1 | PR-AUC | ROC-AUC | NLL | Brier | ECE |
| --- | --- | --- | --- | --- | --- | --- |
| MLP | 0.777599 | 0.839308 | 0.852435 | 0.436951 | 0.142697 | 0.017776 |
| H0 Current Hybrid | 0.774879 | 0.837280 | 0.851260 | 0.439604 | 0.143928 | 0.025288 |
| H1 Tabular Residual Hybrid | 0.777138 | 0.838929 | 0.852624 | 0.438923 | 0.143691 | 0.029090 |

The frozen H1 achieved mean-stage Macro-F1 **0.777138**.
The protocol-matched MLP achieved **0.777599**, a
H1-minus-MLP difference of **-0.000461**. This is classified as a
**PRACTICAL TIE**: MLP is numerically higher by 0.000461, while the paired
95% bootstrap interval crosses zero.

Against H0, H1 improved mean-stage Macro-F1 by **0.002259**, classified
as a **small** final improvement. The evidence does not justify a claim that
deep learning generally outperforms tabular ML.
