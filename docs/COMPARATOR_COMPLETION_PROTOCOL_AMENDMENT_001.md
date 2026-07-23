# Comparator Completion Protocol Amendment 001

Status: **PREREGISTERED BEFORE AFFECTED METRIC RECOMPUTATION**

The base protocol accidentally declared 10 equal-width confidence bins for
ECE. The frozen project calibration contract in
`src/evaluation/calibration.py` uses 15 bins. Replaying the unchanged
Student-Mat CNN-BiLSTM probabilities exposed the mismatch.

ECE is therefore corrected to 15 equal-width confidence bins for every model
and dataset. All affected metric artifacts must be recomputed.

This amendment is a technical consistency correction, not a response to model
performance. It does not change data, targets, features, split membership,
seeds, search spaces, budgets, selected configurations, threshold policy,
probability aggregation, bootstrap, official deep models, recommendation
artifacts, or the Future OULAD lock.

