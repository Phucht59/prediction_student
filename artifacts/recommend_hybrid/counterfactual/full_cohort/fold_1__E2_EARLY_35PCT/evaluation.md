# Counterfactual recommender evaluation

- Status: `PASS`
- Claim boundary: `MODEL_ESTIMATED_RISK_REDUCTION_NOT_CAUSAL_EFFECT`
- Records: `5302`
- Scored coverage: `0.6245`
- Mean top-action risk reduction: `0.118220`
- Median top-action risk reduction: `0.090754`
- Success@0.01: `1.0000`
- Success@0.05: `0.7161`
- Threshold crossing rate: `0.2142`
- Bootstrap mean 95% CI: `[0.11433704388365869, 0.12193999036531125]`

The evaluator measures changes in risk predicted by the frozen Hybrid CNN–BiLSTM under feasible input counterfactuals. Targets and post-cutoff outcomes are not used to rank actions, and the result is not a causal treatment-effect estimate.
