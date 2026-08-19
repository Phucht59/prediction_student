# Five-EBM-C0

**STATUS: PASS** as development control

Features: C0 `risk_probability`, H2 `uncertainty`, `risk_margin`, stage, cutoff-safe evidence. No `action_id`, no Panel B, no Gemini-as-feature.

Honest development slice = 179 portable Panel A queries:

| Metric | Five-EBM-C0 |
|---|---:|
| NDCG@3 | 0.96105 |
| P@1 | 1.000 |
| MRR | 1.000 |
| R@3 | 0.810 |
| pairwise | 0.803 |
| invalid-action | 0.000 |
| unique top1 | 5 |

Mixed LF OOF is higher (0.996) and is **not** used for selection because LFs share features with the ranker.

B0 action+stage prior collapsed to 1 top1 action on the mixed set and is the weak baseline only.
