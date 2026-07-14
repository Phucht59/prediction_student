# Pre-full-run decision

Status: **full V3 benchmark not yet authorized**.

The hardened smoke `model-v3-smoke-20260714b` is valid: 5/5 jobs, 320/320 predictions, no duplicate/missing job or record, valid probabilities and cumulative ordering, train-only regression scaling, valid inverse transform, explicit development/legacy disjointness, and content-valid loader/shape diagnostics. All 145 tests pass, including 5/5 PostgreSQL integration tests.

This establishes implementation readiness only. Authorization still requires human review of the frozen contracts and target-supervision fairness. Smoke scores must not be used to change the search space, acceptance criteria, model registry, or select a preferred family.
