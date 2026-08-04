# Outcome-grounded V2 protocol

Status: `PREREGISTERED_LOCKED_BEFORE_V2_DEVELOPMENT`.

Development uses outer folds 0 and 1. Outer fold 2 is a one-time lockbox and is not used for feature, label, threshold, or model selection. The primary estimand is offline predictive relevance on held-out future OULAD trajectories, not a causal effect, intervention value, or guaranteed grade improvement.

The frozen CNN–BiLSTM risk probability is an allowed OOF feature only. Counterfactual V1 deltas are optional features and are never labels or primary metrics. See `artifacts/recommend_hybrid/outcome_grounded/protocol.json` and `INPUT_AUTHORITY.json`.
