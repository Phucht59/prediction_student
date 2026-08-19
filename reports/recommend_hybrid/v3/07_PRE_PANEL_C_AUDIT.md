# 07 — Pre-Panel-C scientific audit

**STATUS: PASS**

## Invalid-action discrepancy

Legacy unfiltered evaluator (ranks all five actions, including infeasible):
`invalid_action_rate = 0.0012226434`
(`10` / `8179` queries).

Official runtime-equivalent evaluator (hard feasibility → eligible only → rank):
`invalid_action_rate = 0.0`.

Root cause (single class): **A. EVALUATOR_SCOPE_BUG**

All `10` legacy-invalid queries have **zero** feasible actions.
The unfiltered evaluator still ranked the five infeasible rows and counted Top-1 as invalid.
`RecommendationV3Pipeline` filters ineligible actions before `ranker.score` and emits
`NO_FEASIBLE_ACTION` / `INSUFFICIENT_EVIDENCE` with an empty ranking. This is not a
runtime emission of an infeasible action.

Stored `eligible` flags were recomputed with `evaluate_action` on every OOF row:
checked=40895, mismatches=0.

Five-EBM models were **not** refit. Only evaluation semantics were corrected.

## Provenance-separated development metrics (runtime-equivalent)

| Slice | queries | NDCG@3 | P@1 | MRR | R@3 | pairwise | invalid | unique Top-1 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Portable Gemini-supported | 179 | 0.96099 | 1.00000 | 1.00000 | 0.81034 | 0.62617 | 0.0 | 5 |
| LF-only | 7990 | 0.99740 | 1.00000 | 1.00000 | 0.78800 | 0.86201 | 0.0 | 5 |
| Overall development OOF | 8169 | 0.99660 | 1.00000 | 1.00000 | 0.78849 | 0.85286 | 0.0 | 5 |

The large weak-label / LF-only score is **DEVELOPMENT FIT/CONSISTENCY** evidence.
Behavioral labeling functions share the same evidence the EBM sees. It is **not**
confirmatory evidence of real-world recommendation quality.

The portable Gemini-supported slice is the stronger development sanity check.
Panel C is the independent held-out evidence.

## Official development baselines (runtime-equivalent, pre-Panel-C)

| Model | NDCG@3 | P@1 | invalid |
|---|---:|---:|---:|
| B0 action+stage prior | 0.99408 | 1.00000 | 0.0 |
| B1 rule score | 0.99707 | 1.00000 | 0.0 |
| Five-EBM-C0 | 0.99660 | 1.00000 | 0.0 |

## Leakage / runtime / wiring

- Feature leakage pass: `True`
- Pipeline wiring pass: `True`
- Risk-router nullable / no `seed_disagreement` compare: `True`
- Gemini runtime absent: `True`
- Simulator absent: `True`
- Panel B used: false
- Panel C used: false

## Temporal diagnostic (not a model)

{
  "consecutive_stage_pairs": 860,
  "action_switch_rate": 0.5034883720930232,
  "recommendation_persistence": 0.4965116279069768,
  "unsupported_switch_rate": 0.5034883720930232,
  "unsupported_switch_definition": "Diagnostic only. A switch between consecutive intervention stages of the same student-course. unsupported_switch currently equals any switch (no extra temporal model); treat as upper bound."
}

No temporal penalty or temporal model was added.
