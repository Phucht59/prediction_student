# Counterfactual recommender validation

- Status: `PASS`
- Generated at: `2026-08-03T16:37:50.051865Z`
- Claim boundary: `MODEL_ESTIMATED_RISK_REDUCTION_NOT_CAUSAL_EFFECT`
- Counterfactual tests passed: `31`
- Static validation: `PASS`

## Static gates

- `state_actions_known`: `PASS`
- `tensor_actions_known`: `PASS`
- `tensor_actions_cover_planned_oulad`: `PASS`
- `tensor_references_available`: `PASS`
- `mutable_protected_disjoint`: `PASS`
- `frozen_aggregate_preprocessor_api`: `PASS`
- `frozen_static_preprocessor_api`: `PASS`
- `preprocessed_feature_authority_api`: `PASS`
- `evaluation_contract_has_no_outcome_label`: `PASS`
- `outer_fold_evaluator_present`: `PASS`
- `claim_boundary_locked`: `PASS`

## Scientific boundary

The recommender ranks feasible actions by the change in risk estimated by the frozen Hybrid CNN-BiLSTM. This validation does not establish a causal treatment effect, expert agreement, or real-world grade improvement.
