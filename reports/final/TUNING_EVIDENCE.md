# Tuning Evidence

Immutable tuning evidence is retained under `artifacts/final/tuning_evidence`.
It includes trial state (`COMPLETE`/`PRUNED` where recorded), trial number,
objectives, parameters, inner-fold outcomes, screening, focused search,
multi-seed confirmation, selected configuration, runtime and protocol
snapshots.

No trial was fabricated or recomputed during release cleanup. Outer test data
was not used for hyperparameter or candidate selection.
