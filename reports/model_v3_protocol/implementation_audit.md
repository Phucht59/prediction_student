# V3 implementation audit

- Ordered head uses one latent score and two learned ordered thresholds; softplus enforces threshold order.
- Cumulative probabilities are checked for monotonicity before conversion to Low/Medium/High probabilities.
- Multi-task G3 scaler exposes train-fit transform/inverse-transform only; smoke evidence records zero outer-validation rows used to fit it.
- M0 is the exact scikit-learn Small MLP implementation and frozen configuration from Benchmark V2, providing a source/environment control. M1 uses the matched tabular dimensions in PyTorch but necessarily differs in head, loss, optimizer implementation, and framework; this residual fairness limitation must be reviewed before a full run.
- M2/M3 use the same PyTorch tabular backbone and multi-task search space, so their head/loss contrast is controlled.
- Expected-job contract is materialized before the first smoke training call.
- Duplicate expected jobs, metric jobs and record predictions are checked before aggregation.
- Legacy isolation is computed as the explicit intersection of development and legacy record identities.
