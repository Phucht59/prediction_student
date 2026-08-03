# Counterfactual Candidate Release

- Status: **FAIL**
- Candidate state: `CANDIDATE_VALIDATED_PENDING_EXPERT_REVIEW` only when every release gate passes.
- Claim boundary: `MODEL_ESTIMATED_RISK_REDUCTION_NOT_CAUSAL_EFFECT`.

## Steps

- `checkpoint_authority`: **FAIL** (return code `1`)
- `preflight`: **FAIL** (return code `1`)

## Remaining blockers

- Do not merge or promote this candidate until the failed authority gate is repaired and rerun with the real release checkpoint.
- No causal effectiveness, grade improvement, or expert-validation claim is made.
