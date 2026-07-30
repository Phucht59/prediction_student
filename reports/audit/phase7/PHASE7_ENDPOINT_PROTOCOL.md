# Phase 7 Endpoint Protocol

- Endpoint: `F2_MIDDLE_OFFICIAL_SINGLE_CUTOFF`
- Observation cutoff: 50% of module-presentation length
- Availability rule: events with `0 <= date < cutoff`
- Target positive: `Withdrawn` or `Fail`
- Target negative: `Pass` or `Distinction`
- Eligible records: 15,378
- Outer folds: 3, grouped by `id_student`
- Fold sizes: 5,120 / 5,109 / 5,149
- INNER folds: 2
- Final seeds: 42, 1201, 2026, 3407, 7319
- Final H1 runs: 15
- Checkpoint limit: 15 epochs
- Epoch selection: minimum INNER endpoint NLL, then round-half-up median
- Research threshold: pooled INNER OOF Macro-F1
- Outer labels used for tuning or threshold selection: no

H1 uses 47 temporal channels, 165 aggregate features and 13 runtime static
features. Preprocessing is fit only on the applicable training partition.
The endpoint protocol is separate from the frozen 20/35/50/75% early-warning
protocol.
