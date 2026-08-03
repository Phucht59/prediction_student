# Counterfactual recommender evaluation

- Status: `PASS`
- Claim boundary: `MODEL_ESTIMATED_RISK_REDUCTION_NOT_CAUSAL_EFFECT`
- Records: `4875`
- Scored coverage: `0.7054`
- Mean top-action risk reduction: `0.112806`
- Median top-action risk reduction: `0.067147`
- Success@0.01: `1.0000`
- Success@0.05: `0.5856`
- Threshold crossing rate: `0.1810`
- Bootstrap mean 95% CI: `[0.10892967053090027, 0.11718697013191534]`

The evaluator measures changes in risk predicted by the frozen Hybrid CNN–BiLSTM under feasible input counterfactuals. Targets and post-cutoff outcomes are not used to rank actions, and the result is not a causal treatment-effect estimate.
