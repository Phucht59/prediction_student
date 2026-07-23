# Comparator Completion Protocol Amendment 002

Status: **PREREGISTERED BEFORE AFFECTED METRIC RECOMPUTATION**

Frozen OULAD replay showed that the repository intentionally uses two ECE
contracts:

- Student-Mat and Student-Por use 15 equal-width bins of predicted-class
  confidence (`src/evaluation/calibration.py`).
- Binary OULAD uses 10 equal-width bins of the positive At-risk probability
  (`src/studies/v5/common/metrics.py`).

Amendment 001 correctly restored the UCI definition but incorrectly described
the change as applying to every dataset. Amendment 002 narrows and completes
that correction. All affected metric artifacts must be recomputed.

This is a replay-driven technical consistency correction only. No data,
features, split, seed, search, selected configuration, threshold, probability,
deep checkpoint, recommendation artifact, or Future-lock state changes.

