# Counterfactual Shortcut Diagnostic

- Status: **PASS**
- Records / scored: `1200` / `780`
- Top-action concentration: `0.5833`
- Unique top actions: `5`
- Top risk-reduction standard deviation: `0.109366`

| Gate | Result |
|---|---|
| action_identity_not_over_80_percent | PASS |
| multiple_top_actions_observed | PASS |
| risk_reduction_not_constant | PASS |
| risk_reduction_not_mostly_zero | PASS |

This is a trace-level shortcut audit. It reports action/stage/baseline variation from frozen-model evaluation traces; it does not claim a causal effect and does not replace an input-level model permutation experiment.
