# Recommendation source

## Final scientifically validated boundary

The final validated module is `src/recommend_hybrid/final`:

```text
external deterministic policy or instructor eligibility
→ frozen residual CNN–BiLSTM representation
→ integrated conditional action head
→ ranked eligible actions
→ downstream prerequisite/conflict/workload/safety constraints
```

Public import:

```python
from src.recommend_hybrid.final import ConditionalHybridActionRanker
```

The ranker is fail-closed:

- external eligibility must be explicit;
- scores must carry authority `integrated_conditional_action_head`;
- the model output must contain the five fixed scientific action slots;
- caller-authored per-action scores are rejected;
- only `offline_evaluation` execution is accepted;
- production runtime remains unauthorized.

Validated held-out result: ranking-only Precision@1 `0.9374`. This is not
end-to-end recommendation accuracy. Automatic issuance remains not validated
(V4 end-to-end Precision@1 `0.6589`, positive-group coverage `0.4980`).

## Legacy research APIs

`pipeline.py`, `common/`, `uci/`, and `oulad/` are retained for reproducibility
and policy/constraint support. They are not the validated final automatic
recommendation runtime and must not be described as production-authorized.

Do not change frozen prediction authority, final scientific values, or policy
constraints without a new preregistered release protocol.
