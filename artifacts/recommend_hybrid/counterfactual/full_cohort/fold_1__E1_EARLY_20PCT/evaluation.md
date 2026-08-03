# Counterfactual recommender evaluation

- Status: `PASS`
- Claim boundary: `MODEL_ESTIMATED_RISK_REDUCTION_NOT_CAUSAL_EFFECT`
- Records: `5514`
- Scored coverage: `0.5550`
- Mean top-action risk reduction: `0.101365`
- Median top-action risk reduction: `0.081047`
- Success@0.01: `1.0000`
- Success@0.05: `0.6873`
- Threshold crossing rate: `0.1561`
- Bootstrap mean 95% CI: `[0.09843941246749821, 0.1045458732575811]`

The evaluator measures changes in risk predicted by the frozen Hybrid CNN–BiLSTM under feasible input counterfactuals. Targets and post-cutoff outcomes are not used to rank actions, and the result is not a causal treatment-effect estimate.
