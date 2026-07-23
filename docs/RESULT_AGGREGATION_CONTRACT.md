# Result Aggregation Contract

## Accepted scopes

Final values may originate only from:

- `FINAL_OUTER_OOF`
- `FINAL_PROBABILITY_ENSEMBLE`

The following are forbidden as final results: `INNER_OOF_SCREENING`, `SEED_MEAN`, `BEST_SEED`, and `BEST_FOLD`.

All three datasets use the exact same nine-row model catalog and order. Every
numeric metric records its prediction artifact, SHA-256 checksum, protocol
hash, split-manifest hash, feature-contract hash, and calculation method.
Applicable missing model metrics are prohibited in `final_results_v2`.

Frozen deep predictions and preregistered completion-comparator predictions are
both validated record by record before aggregation. Their provenance is
distinguished as `frozen_existing`, `derived_from_frozen_prediction`, or
`newly_trained_comparator`; no screening metric is promoted to a final result.

Macro-F1 must equal the unweighted mean of class F1 values. Overall Precision/Recall are macro averages. Top-k requires a frozen probability, stable record IDs, descending sorting, record-ID tie breaking and upward budget rounding.

The per-class schema contains only class, precision, recall, F1, and support.
Macro-F1 appears only in the overall table. UCI paired bootstrap uses records;
OULAD paired bootstrap resamples complete `id_student` groups.
