# Final H1 Uncertainty

Paired bootstrap used **5,000** replicates, resampling by `id_student`, with
predictions aligned on the same held-out observations.

| Comparison | Population delta | Bootstrap mean | 95% CI | Crosses zero |
| --- | --- | --- | --- | --- |
| H1 − MLP | -0.000430 | -0.000420 | [-0.002778, 0.001858] | YES |
| H1 − H0 | 0.001834 | 0.001833 | [-0.000253, 0.003853] | YES |

Neither comparison supports robust superiority at the paired 95% interval.
Numerical direction remains reportable separately from robust evidence.
