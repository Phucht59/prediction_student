# Validation

- Phase 1–8 audit plus release regression tests: 152 passed
- Final comparator validator: PASS
- OULAD validator: PASS
- UCI regression validator: PASS
- Ruff on changed Python files: PASS
- `compileall` on changed Python files: PASS
- H0 metric reproduced: PASS
- H1 metric reproduced: PASS
- Record/target/fold identity: PASS
- Same Macro-F1 implementation: PASS
- Score/future-feature audit: PASS with documented H0 proxy caveat
- Train-only preprocessing scope: PASS
- Early-warning checksums unchanged: PASS
- New training runs: 0
- Optuna trials: 0
- New outer evaluations: 0

The diagnostic outer-oracle threshold is explicitly non-selective and did not
mutate configuration or predictions.
