# V25 Elite Ensemble Report

**Strategy**: Top-K selection from diverse member pool, TTA x5
**Key insight**: Oversampling (ADASYN/SMOTE) + short training is critical
**Architecture**: SE-CNN-BiLSTM + Deep Residual MLP with diverse configs

## Final Results

| Dataset | Members | Best-K | **Best F1** | V18 Best | Paper F1 | Beat V18? |
| --- | --- | --- | --- | --- | --- | --- |
| xapi | 80 | Top-1 | **0.7832** | 0.7841 | 0.8447 | ❌ |

## Top-K Sweep (F1 by ensemble size)

| Dataset | Top-1 | Top-3 | Top-5 | Top-7 | Top-10 | Top-15 |
| --- | --- | --- | --- | --- | --- | --- |
| xapi | 0.7832 | 0.7704 | 0.7314 | 0.7706 | 0.7037 | 0.7579 |

## Top-20 Members per Dataset

| Dataset | Member | Val F1 | Ep | Feat | Sampling | Model |
| --- | --- | --- | --- | --- | --- | --- |
| xapi | seed123 | 0.7762 | 19 | full | smote | secnn_bilstm |
| xapi | seed2048 | 0.7722 | 41 | full | smote | deep_res_mlp |
| xapi | seed2024 | 0.7702 | 50 | full | adasyn | deep_res_mlp |
| xapi | seed777 | 0.7685 | 21 | full | adasyn | deep_res_mlp |
| xapi | seed42 | 0.7595 | 53 | full | smote | deep_res_mlp |
| xapi | seed777 | 0.7591 | 18 | full | smote | deep_res_mlp |
| xapi | seed123 | 0.7574 | 29 | full | adasyn | secnn_bilstm |
| xapi | seed42 | 0.7571 | 39 | full | adasyn | deep_res_mlp |
| xapi | seed512 | 0.7567 | 24 | full | class_weight | deep_res_mlp |
| xapi | seed1234 | 0.7564 | 23 | full | adasyn | deep_res_mlp |
| xapi | seed512 | 0.7504 | 23 | behavior8 | smote | deep_res_mlp |
| xapi | seed7 | 0.7492 | 50 | behavior8 | smote | deep_res_mlp |
| xapi | seed42 | 0.7486 | 11 | behavior8 | smote | deep_res_mlp |
| xapi | seed999 | 0.7481 | 18 | behavior8 | none | secnn_bilstm |
| xapi | seed123 | 0.7478 | 37 | full | smote | deep_res_mlp |
| xapi | seed314 | 0.7474 | 7 | behavior8 | none | secnn_bilstm |
| xapi | seed777 | 0.7474 | 20 | behavior8 | smote | deep_res_mlp |
| xapi | seed31 | 0.7462 | 50 | full | none | secnn_bilstm |
| xapi | seed256 | 0.7456 | 12 | full | adasyn | deep_res_mlp |
| xapi | seed777 | 0.7428 | 16 | paper | smote | deep_res_mlp |
