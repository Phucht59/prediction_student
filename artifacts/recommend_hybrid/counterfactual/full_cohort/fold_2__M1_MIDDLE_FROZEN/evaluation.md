# Counterfactual recommender evaluation

- Status: `PASS`
- Claim boundary: `MODEL_ESTIMATED_RISK_REDUCTION_NOT_CAUSAL_EFFECT`
- Records: `5149`
- Scored coverage: `0.7687`
- Mean top-action risk reduction: `0.110529`
- Median top-action risk reduction: `0.066447`
- Success@0.01: `1.0000`
- Success@0.05: `0.6094`
- Threshold crossing rate: `0.2381`
- Bootstrap mean 95% CI: `[0.10697306239944065, 0.11404867145058027]`

The evaluator measures changes in risk predicted by the frozen Hybrid CNN–BiLSTM under feasible input counterfactuals. Targets and post-cutoff outcomes are not used to rank actions, and the result is not a causal treatment-effect estimate.
