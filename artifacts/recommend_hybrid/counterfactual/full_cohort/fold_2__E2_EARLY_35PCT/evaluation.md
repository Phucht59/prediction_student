# Counterfactual recommender evaluation

- Status: `PASS`
- Claim boundary: `MODEL_ESTIMATED_RISK_REDUCTION_NOT_CAUSAL_EFFECT`
- Records: `5345`
- Scored coverage: `0.6200`
- Mean top-action risk reduction: `0.112086`
- Median top-action risk reduction: `0.083098`
- Success@0.01: `1.0000`
- Success@0.05: `0.6611`
- Threshold crossing rate: `0.1896`
- Bootstrap mean 95% CI: `[0.1087531284903225, 0.11541194372771642]`

The evaluator measures changes in risk predicted by the frozen Hybrid CNN–BiLSTM under feasible input counterfactuals. Targets and post-cutoff outcomes are not used to rank actions, and the result is not a causal treatment-effect estimate.
