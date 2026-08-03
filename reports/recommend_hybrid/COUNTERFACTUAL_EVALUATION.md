# Counterfactual recommender evaluation

- Status: `PASS`
- Claim boundary: `MODEL_ESTIMATED_RISK_REDUCTION_NOT_CAUSAL_EFFECT`
- Records: `1200`
- Scored coverage: `0.6500`
- Mean top-action risk reduction: `0.102294`
- Median top-action risk reduction: `0.064228`
- Success@0.01: `1.0000`
- Success@0.05: `0.5987`
- Threshold crossing rate: `0.1852`
- Bootstrap mean 95% CI: `[0.09497193479570203, 0.10963785617612302]`

The evaluator measures changes in risk predicted by the frozen Hybrid CNN–BiLSTM under feasible input counterfactuals. Targets and post-cutoff outcomes are not used to rank actions, and the result is not a causal treatment-effect estimate.
