# Final database roundtrip

PostgreSQL was reachable. Migration `001_recommendation_runtime.sql` applied additively. Existing catalog rows were not deleted.

## Smoke case

CLI inference for enrollment `000095ce8af9578334f5d8af` stage `20pct`:

| Field | Source | Stored |
|---|---|---|
| risk_probability | 0.04911900435884794 | 0.04911900435884794 |
| plan_status | REVIEW | REVIEW |
| A1 feasibility | INFEASIBLE | INFEASIBLE |
| A4 feasibility | FEASIBLE | FEASIBLE |
| A5 release_status | REVIEW_REQUIRED | REVIEW_REQUIRED |
| A5 relevance | 1.224679083596255 | 1.224679083596255 |
| explanations | 5 local EBM payloads | 5 |
| bundle_version | recommendation.final_freeze.v1 | recommendation.final_freeze.v1 |

Re-running the same request reused `run_id=614e4373-ca0e-481b-b1b1-e461abe17922` (idempotent upsert).

## 10-case batch

`load_final_results_to_postgres.py --limit 10` inserted 10 additional runs (11 total including the smoke case). No duplicate unique keys.

## Full load

`load_final_results_to_postgres.py` completed:

| Entity | Count |
|---|---:|
| state_snapshots | 100061 |
| runs | 100061 |
| scores | 500305 |
| explanations | 500305 |
| plans | 100061 |

Catalog students/courses/enrollments were reused (29447 / 24 / 33621). No existing user rows were deleted.
