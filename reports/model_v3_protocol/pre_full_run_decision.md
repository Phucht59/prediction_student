# Pre-full-run decision

Status: **full V3 benchmark not yet authorized**.

The authoritative hardened smoke `model-v3-smoke-20260714c` is valid: 5/5 jobs, 320/320 predictions, no duplicate/missing job or record, valid probabilities and cumulative ordering, train-only regression scaling, valid inverse transform, explicit development/legacy disjointness, and content-valid loader/shape diagnostics. M0 uses the exact Benchmark V2 scikit-learn Small MLP implementation and frozen control configuration. All 145 tests pass, including 5/5 PostgreSQL integration tests.

This establishes implementation readiness only. Authorization still requires human review of the frozen contracts and target-supervision fairness. In particular, M0 is an exact source/environment control implemented in scikit-learn while M1 is implemented in PyTorch; backbone dimensions are controlled, but optimizer/training implementation is not identical. Smoke scores must not be used to change the search space, acceptance criteria, model registry, or select a preferred family.
