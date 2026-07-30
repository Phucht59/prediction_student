# Phase 2 — Validation

## Scientific safeguards

- Outer labels used for epoch/threshold/model selection: **NO**
- Optuna executed: **NO**
- Official final files changed since Phase 1: **NO**
- Same checkpoint across OULAD stages: **PASS** (frozen 600 mappings)
- Future mask and preprocessing isolation: **PASS**
- OULAD group-disjoint inner splits: **PASS**

## UCI quasi-group warning

UCI has no true student identifier. The retained proxy is the Phase 1
quasi-identity built from demographic/family attributes. Student-Mat has
4 folds with nonzero proxy
overlap (maximum 4);
Student-Por has 5 (maximum
6). Record
intersections remain zero. This is **not confirmed leakage**, and Phase 2 does
not change the frozen split protocol. A new-protocol sensitivity analysis must
be separate.

## Executed validation

- Phase 2 + Phase 1 audit tests and relevant unified/final release tests:
  **82 passed**.
- Ruff on every changed Python file: **PASS**.
- `compileall` for source, scripts, and audit tests: **PASS**.
- `project.py final validate`: **FINAL_COMPARATOR_COMPLETION_PASS**.
- `project.py pipeline uci validate`: **PASS**.
- `project.py pipeline oulad validate`: **PASS**.
- Official final checksum freeze: **unchanged=true**.
