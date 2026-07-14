# V3 implementation audit

- Ordered head uses one latent score and two learned ordered thresholds; softplus enforces threshold order.
- Cumulative probabilities are checked for monotonicity before conversion to Low/Medium/High probabilities.
- Multi-task G3 scaler exposes train-fit transform/inverse-transform only; smoke evidence records zero outer-validation rows used to fit it.
- M0/M1 and M2/M3 use matched tabular backbones. Head-only parameter differences are reported separately.
- Expected-job contract is materialized before the first smoke training call.
- Duplicate expected jobs, metric jobs and record predictions are checked before aggregation.
- Legacy isolation is computed as the explicit intersection of development and legacy record identities.
