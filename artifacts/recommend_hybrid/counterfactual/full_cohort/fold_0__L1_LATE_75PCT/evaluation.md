# Counterfactual recommender evaluation

- Status: `PASS`
- Claim boundary: `MODEL_ESTIMATED_RISK_REDUCTION_NOT_CAUSAL_EFFECT`
- Records: `4863`
- Scored coverage: `0.7162`
- Mean top-action risk reduction: `0.119031`
- Median top-action risk reduction: `0.068787`
- Success@0.01: `1.0000`
- Success@0.05: `0.6018`
- Threshold crossing rate: `0.1934`
- Bootstrap mean 95% CI: `[0.11436890891906619, 0.12405930927628665]`

The evaluator measures changes in risk predicted by the frozen Hybrid CNN–BiLSTM under feasible input counterfactuals. Targets and post-cutoff outcomes are not used to rank actions, and the result is not a causal treatment-effect estimate.
