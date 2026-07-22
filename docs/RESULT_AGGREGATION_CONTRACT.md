# Result Aggregation Contract

## Accepted scopes

Final values may originate only from:

- `FINAL_OUTER_OOF`
- `FINAL_PROBABILITY_ENSEMBLE`

The following are forbidden as final results: `INNER_OOF_SCREENING`, `SEED_MEAN`, `BEST_SEED`, and `BEST_FOLD`.

All three datasets use the exact same nine-row model catalog and order. Every numeric metric records a source artifact, its SHA-256 checksum and whether it was loaded or recomputed from frozen predictions/confusion matrices. Missing evidence is represented as JSON `null` with status `N/A` and a reason; it is never estimated.

Macro-F1 must equal the unweighted mean of class F1 values. Overall Precision/Recall are macro averages. Top-k requires a frozen probability, stable record IDs, descending sorting, record-ID tie breaking and upward budget rounding.
