# Counterfactual recommender evaluation

- Status: `PASS`
- Claim boundary: `MODEL_ESTIMATED_RISK_REDUCTION_NOT_CAUSAL_EFFECT`
- Records: `5120`
- Scored coverage: `0.7611`
- Mean top-action risk reduction: `0.103804`
- Median top-action risk reduction: `0.063767`
- Success@0.01: `1.0000`
- Success@0.05: `0.5925`
- Threshold crossing rate: `0.1931`
- Bootstrap mean 95% CI: `[0.10059445891458821, 0.10734887548558861]`

The evaluator measures changes in risk predicted by the frozen Hybrid CNN–BiLSTM under feasible input counterfactuals. Targets and post-cutoff outcomes are not used to rank actions, and the result is not a causal treatment-effect estimate.
