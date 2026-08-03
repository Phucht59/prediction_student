# Counterfactual recommender evaluation

- Status: `PASS`
- Claim boundary: `MODEL_ESTIMATED_RISK_REDUCTION_NOT_CAUSAL_EFFECT`
- Records: `5109`
- Scored coverage: `0.7320`
- Mean top-action risk reduction: `0.111374`
- Median top-action risk reduction: `0.069144`
- Success@0.01: `1.0000`
- Success@0.05: `0.6235`
- Threshold crossing rate: `0.2113`
- Bootstrap mean 95% CI: `[0.10760630869152234, 0.1151918692644287]`

The evaluator measures changes in risk predicted by the frozen Hybrid CNN–BiLSTM under feasible input counterfactuals. Targets and post-cutoff outcomes are not used to rank actions, and the result is not a causal treatment-effect estimate.
