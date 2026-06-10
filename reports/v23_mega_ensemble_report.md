# V23 Mega Ensemble Report

**Strict Protocol**: Fixed 15% test set, K-fold CV on 85%, Multi-feature ensemble
**Architecture**: MultiScale CNN + 2-layer BiLSTM + Multi-Head Attention
**Training**: Focal Loss + Mixup + CosineAnnealing + Optuna-tuned hyperparams

## Final Results

| Dataset | Members | Uniform F1 | Weighted F1 | Best F1 | Paper F1 | Gap | Target |
| --- | --- | --- | --- | --- | --- | --- | --- |
| student-mat | 30 | 0.6953 | 0.6953 | 0.6953 | 0.9400 | -0.2447 | ❌ |
| student-por | 30 | 0.6895 | 0.6895 | 0.6895 | 0.9000 | -0.2105 | ❌ |
| xapi | 10 | 0.7135 | 0.7135 | 0.7135 | 0.8447 | -0.1312 | ❌ |
