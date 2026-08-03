# Counterfactual recommender evaluation

- Status: `PASS`
- Claim boundary: `MODEL_ESTIMATED_RISK_REDUCTION_NOT_CAUSAL_EFFECT`
- Records: `5539`
- Scored coverage: `0.5763`
- Mean top-action risk reduction: `0.103838`
- Median top-action risk reduction: `0.075691`
- Success@0.01: `1.0000`
- Success@0.05: `0.6598`
- Threshold crossing rate: `0.1779`
- Bootstrap mean 95% CI: `[0.10101937646615812, 0.10688539141839262]`

The evaluator measures changes in risk predicted by the frozen Hybrid CNN–BiLSTM under feasible input counterfactuals. Targets and post-cutoff outcomes are not used to rank actions, and the result is not a causal treatment-effect estimate.
