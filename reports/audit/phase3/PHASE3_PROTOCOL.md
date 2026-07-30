# Phase 3 — Protocol

- Dataset/model: unified stage-aware OULAD CNN-BiLSTM only.
- Outer folds: 3; inner folds: 2 grouped folds.
- Outer labels accessible to runner: no.
- Shared estimator/checkpoint across 20/35/50/75%.
- Primary objective: maximize pooled-inner-OOF mean-stage Macro-F1.
- Checkpoint policy: minimize mean-stage validation NLL.
- Epoch cap: 15; patience: 5.
- Inner→refit epoch: round-half-up median.
- Research threshold: pooled inner OOF Macro-F1.
- Operational threshold: excluded from Optuna.
- Sampler: TPE, seeds 42/43/44, six startup trials.
- Pruner: MedianPruner, warm-up 3 epochs; intermediate signal = negative NLL
  because the study direction is maximize.
- Budget: 24 scheduled trials per fold, no automatic extension.
- Search training seed: 42.
- Stability seeds: [1201, 2026].
- GPU concurrency: one; FP32; AMP disabled.
