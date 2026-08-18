# Root cause

Classification: **I — MULTIPLE FACTORS**.

1. **Endpoint feature-authority and score-signal difference** — HIGH; CONFIRMED_DIFFERENCE_CAUSAL_SHARE_INCONCLUSIVE.
2. **Phase7 H1 is not an H0-plus-residual reproduction** — HIGH; CONFIRMED_DIFFERENCE.
3. **Confirmed ranking and probability-quality deficit** — HIGH; CONFIRMED_OUTCOME.
4. **Train-time preprocessing changed materially** — MEDIUM; CONFIRMED_DIFFERENCE.
5. **Temporal topology and fusion input contract changed** — MEDIUM; CONFIRMED_DIFFERENCE_CAUSAL_SHARE_INCONCLUSIVE.
6. **Endpoint pretraining present only in H0** — MEDIUM; CONFIRMED_DIFFERENCE.
7. **Endpoint training and checkpoint recipe changed** — MEDIUM; CONFIRMED_DIFFERENCE.
8. **Threshold difference is not the main cause** — LOW; CLEARED_AS_PRIMARY_CAUSE.
9. **Population, target, cutoff, and outer folds match** — LOW; NOT_A_CAUSE.

The strongest diagnosis is not “H1 residual expert is bad.” The endpoint
experiment changed too many upstream contracts to isolate the residual expert.
The final performance deficit itself is confirmed, but the exact fraction
caused by feature authority, preprocessing, architecture and training cannot be
separated without a new development-only factorial study.
