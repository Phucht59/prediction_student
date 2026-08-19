# Recommendation V3 — final report

**FINAL_RECOMMENDATION_V3_READY = false**

**Blocker:** Panel C complete coverage failed. Authentic Gemini pass 1 reviewed 501 / 632 cases. The remaining 131 cases hit HTTP 429 after the frozen 3-attempt retry policy. The provider daily free-tier cap is 500 `generate_content_free_tier_requests`. No reviews were fabricated.

## Runtime (unchanged candidate)

```text
Phase4 Hybrid C0
    → risk_probability + STOP threshold + H2(p)
    → cutoff-safe OULAD 20/35/50/75 evidence
    → C0-aligned risk router
    → hard feasibility
    → Five-EBM-C0
    → simple safety router
    → RECOMMEND Top-1 or HUMAN_REVIEW Top-3
    → deterministic personalized plan
```

Prediction authority was not changed. No prediction HPO. No challenger. No Panel B. No Gemini at runtime. No simulator core. No UCI recommendation. OULAD 100% cannot intervene.

## Development evidence (not held-out)

Runtime-equivalent evaluation (feasible actions only):

| Slice | queries | NDCG@3 | P@1 | invalid |
|---|---:|---:|---:|---:|
| Portable Gemini-supported | 179 | 0.96099 | 1.000 | 0.0 |
| LF-only | 8000 | 0.99740 | 1.000 | 0.0 |
| Overall development OOF | 8169 | 0.99660 | 1.000 | 0.0 |

The large LF-only / overall numbers are **development fit/consistency**. Behavioral labeling functions share the EBM evidence. They are not confirmatory.

Official development baselines (runtime-equivalent):

| Model | NDCG@3 | invalid |
|---|---:|---:|
| B0 action+stage prior | 0.99408 | 0.0 |
| B1 rule score | 0.99707 | 0.0 |
| Five-EBM-C0 | 0.99660 | 0.0 |

B1 beating Five-EBM on the weak-label OOF is expected circularity. The frozen candidate stays Five-EBM-C0.

## Pre-Panel-C audit

`PRE_PANEL_C_AUDIT = PASS`

Legacy unfiltered `invalid_action_rate = 0.0012226` (10 / 8179). All 10 queries had **zero** feasible actions. The evaluator ranked infeasible rows. Runtime filters first and emits an empty abstain route. Root cause: **A. EVALUATOR_SCOPE_BUG**. Official runtime-equivalent invalid-action rate is 0. Five-EBM was not refit.

## Final held-out evidence

**None claimed.**

Panel C was opened with authentic `gemini-3.5-flash-lite` and the frozen blinded V3 prompt. 1910 authentic review records exist for 501 cases. Coverage is incomplete, so one-shot evaluation was not run.

Historical Panel B is Recommendation V2 held-out evidence. It was not used for V3 tuning or evaluation.

## Allowed vs forbidden claims

Allowed only after a complete authentic Panel C evaluation: predictive action-relevance ranking, feasibility-valid recommendations, explainable logic, C0-guided selection, deterministic plans.

Not allowed now or later without prospective outcome evidence: causal intervention effect, guaranteed academic improvement, real-world outcome improvement.
