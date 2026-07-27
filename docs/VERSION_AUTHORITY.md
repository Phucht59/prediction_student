# Version authority

| Version | Authoritative role | Supersession boundary |
|---|---|---|
| V5 | Historical controlled model evidence | Retained; not the canonical final release |
| V5.1 | Canonical UCI prediction evidence and frozen OULAD reference | Prediction artifacts remain immutable |
| V5.2–V5.4 | Diagnostic/extension evidence | Do not silently replace canonical V5.1/V6 results |
| V6 | Canonical integrated OULAD prediction/risk-profile evidence | Prediction OOF, checkpoints and model registry remain frozen |
| V6.1 | OULAD architecture diagnosis | Negative development gate retained; no new final model |
| V6.2 | Recommendation scientific validation, expert package, claim/database audit | Evaluation-only; no training, model selection, outer-test opening or Future OULAD |

The canonical result source for cross-model classification comparison is
`artifacts/final/final_results.csv`. V6.2 may reference it, but never rewrite
its metrics.
