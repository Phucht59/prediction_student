# Counterfactual recommender evaluation

- Status: `PASS`
- Claim boundary: `MODEL_ESTIMATED_RISK_REDUCTION_NOT_CAUSAL_EFFECT`
- Records: `4897`
- Scored coverage: `0.7125`
- Mean top-action risk reduction: `0.121348`
- Median top-action risk reduction: `0.067801`
- Success@0.01: `1.0000`
- Success@0.05: `0.6013`
- Threshold crossing rate: `0.2180`
- Bootstrap mean 95% CI: `[0.1163810761251179, 0.1262848549941408]`

The evaluator measures changes in risk predicted by the frozen Hybrid CNN–BiLSTM under feasible input counterfactuals. Targets and post-cutoff outcomes are not used to rank actions, and the result is not a causal treatment-effect estimate.
