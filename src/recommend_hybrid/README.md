# Recommendation source

## Final validated module

Use `src/recommend_hybrid/final/` for the final recommendation component.
`ConditionalHybridActionRanker` ranks policy-authorized scientific actions by
identity using output from the integrated conditional action head over the
frozen residual CNN–BiLSTM representation.

```python
from src.recommend_hybrid.final import ConditionalHybridActionRanker
```

The caller must explicitly provide external eligibility and integrated-head
output. The final API fails closed when eligibility or model scores are absent.
The fixed model action order is defined in `final/actions.py`; policy catalog
aliases are mapped to those trained action identities before ranking.

## Scientific boundary

- Conditional action ranking: validated offline.
- End-to-end recommendation issuance: not validated.
- Runtime authorization: false.
- Causal effect and guaranteed grade improvement: not claimed.

The legacy `pipeline.py`, `uci/`, `oulad/`, and `common/` modules are retained
for compatibility and constraint/policy utilities. They are not the validated
final action-ranking entry point and must not be presented as having 93.74%
end-to-end accuracy.
