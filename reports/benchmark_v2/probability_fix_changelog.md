# Probability fallback fix changelog

- Changed `scripts/run_benchmark_v2.py:probs_from_pred` to call the central deterministic `hard_label_probabilities` helper.
- Added `hard_label_probabilities` and `validate_probability_matrix` in `src/evaluation/protocol.py`. The validator requires float64-compatible three-class Low/Medium/High rows, finite values in `[0, 1]`, sum-to-one within `1e-6`, and label/argmax agreement. The `1e-6` tolerance is strictly for float32 neural softmax numerical precision and does not admit the prior `0.001` violation.
- Added a pre-artifact validation gate in the benchmark runner; invalid probability rows now stop a run before predictions are written as complete.
- Added regression tests for all three hard labels, multi-record batches, JSON serialization round-trip, and rejection of the historical `[0.999, 0.001, 0.001]` invalid vector.

This changes only probability representation for hard-label fallback baselines. It does not change data, folds, features, target mapping, tuning budget, seeds, training, early stopping, resampling, class weights, architecture, loss, or the ranking metric.
