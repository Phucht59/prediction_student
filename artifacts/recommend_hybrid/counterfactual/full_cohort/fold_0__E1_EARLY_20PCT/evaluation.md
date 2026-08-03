# Counterfactual recommender evaluation

- Status: `PASS`
- Claim boundary: `MODEL_ESTIMATED_RISK_REDUCTION_NOT_CAUSAL_EFFECT`
- Records: `5513`
- Scored coverage: `0.5846`
- Mean top-action risk reduction: `0.105300`
- Median top-action risk reduction: `0.073273`
- Success@0.01: `1.0000`
- Success@0.05: `0.6326`
- Threshold crossing rate: `0.1763`
- Bootstrap mean 95% CI: `[0.10205573730148766, 0.10835589741021764]`

The evaluator measures changes in risk predicted by the frozen Hybrid CNN–BiLSTM under feasible input counterfactuals. Targets and post-cutoff outcomes are not used to rank actions, and the result is not a causal treatment-effect estimate.
