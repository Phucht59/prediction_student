# Counterfactual recommender evaluation

- Status: `PASS`
- Claim boundary: `MODEL_ESTIMATED_RISK_REDUCTION_NOT_CAUSAL_EFFECT`
- Records: `62525`
- Scored coverage: `0.6633`
- Mean top-action risk reduction: `0.111392`
- Median top-action risk reduction: `0.073885`
- Success@0.01: `1.0000`
- Success@0.05: `0.6363`
- Threshold crossing rate: `0.2006`
- Bootstrap mean 95% CI: `[0.11034337363786825, 0.11247858550161341]`

The evaluator measures changes in risk predicted by the frozen Hybrid CNN–BiLSTM under feasible input counterfactuals. Targets and post-cutoff outcomes are not used to rank actions, and the result is not a causal treatment-effect estimate.
