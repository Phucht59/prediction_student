# Recommendation V — final report

**FINAL_RECOMMENDATION_V3_READY = true**

## Runtime authority

```text
Hybrid CNN–BiLSTM
    → risk_probability + STOP threshold + H2(p)
    → cutoff-safe OULAD 20/35/50/75 evidence
    → risk router
    → hard feasibility
    → Recommendation V
    → simple safety router
    → RECOMMEND Top-1 or HUMAN_REVIEW Top-3
    → deterministic personalized plan
```

Prediction authority was not changed. No prediction HPO. No challenger. No Panel B for tuning. No Gemini at runtime. No simulator core. No UCI recommendation. OULAD 100% cannot intervene.

## Development evidence (not held-out)

Runtime-equivalent evaluation (feasible actions only):

| Slice | queries | NDCG@3 | P@1 | invalid |
|---|---:|---:|---:|---:|
| Portable Gemini-supported | 179 | 0.96099 | 1.000 | 0.0 |
| LF-only | 8000 | 0.99740 | 1.000 | 0.0 |
| Overall development OOF | 8169 | 0.99660 | 1.000 | 0.0 |

The large LF-only / overall numbers are **development fit/consistency**. They are not confirmatory.

Official development baselines (runtime-equivalent): B0 NDCG@3 = 0.99408; B1 = 0.99707; Recommendation V = 0.99660. B1 beating Five-EBM on weak labels is expected circularity. The frozen candidate stayed Recommendation V.

`PRE_PANEL_C_AUDIT = PASS`. Legacy invalid-action rate 0.0012226 was **A. EVALUATOR_SCOPE_BUG**. Official runtime-equivalent invalid-action rate is 0.

## Final held-out evidence (Panel C only)

632 student-stage cases, 150 students, disjoint from portable Panel A. 2398 authentic reviews. Primary metrics use feasible candidates only.

| Model | NDCG@3 | P@1 | MRR | R@3 | pairwise | invalid |
|---|---:|---:|---:|---:|---:|---:|
| Recommendation V | 0.88785 | 0.99206 | 0.99603 | 0.79947 | 0.53849 | 0.0 |
| B0 action+stage | 0.81889 | 0.99365 | 0.99683 | 0.78981 | 0.25377 | 0.0 |
| B1 rule score | 0.86649 | 0.99683 | 0.99841 | 0.80357 | 0.45402 | 0.0 |

Exact-best Top-1 agreement: 0.407.

Bootstrap Recommendation V minus B1 (best baseline) NDCG@3: mean 0.02131, 95% CI [0.01440, 0.02815], P(diff>0)=1.0, 2000 iterations, seed 2026.

Pipeline on the same 632 cases: RECOMMEND 94, HUMAN_REVIEW 175, INSUFFICIENT_EVIDENCE 363, NO_FEASIBLE_ACTION 0. Invalid-action rate 0. Unique Top-1 actions: 5.

Panel C reviewers did not see risk, uncertainty, model identity, ranks, or outcomes.

## Historical Panel B

Historical recommendation held-out (previous release). Not used for Recommendation V tuning or final evaluation.

## Claim boundary

Allowed: predictive action-relevance ranking on fresh Panel C reviews; feasibility-valid recommendations; explainable Five-EBM logic; C0-guided routing; deterministic personalized plans.

Not allowed: causal intervention effectiveness, guaranteed academic improvement, treatment effect, real-world outcome improvement without prospective evidence.
