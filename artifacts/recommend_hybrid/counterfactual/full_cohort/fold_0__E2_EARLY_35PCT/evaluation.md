# Counterfactual recommender evaluation

- Status: `PASS`
- Claim boundary: `MODEL_ESTIMATED_RISK_REDUCTION_NOT_CAUSAL_EFFECT`
- Records: `5299`
- Scored coverage: `0.6352`
- Mean top-action risk reduction: `0.116262`
- Median top-action risk reduction: `0.085214`
- Success@0.01: `1.0000`
- Success@0.05: `0.6887`
- Threshold crossing rate: `0.2780`
- Bootstrap mean 95% CI: `[0.11298238225126295, 0.11951084549113249]`

The evaluator measures changes in risk predicted by the frozen Hybrid CNN–BiLSTM under feasible input counterfactuals. Targets and post-cutoff outcomes are not used to rank actions, and the result is not a causal treatment-effect estimate.
