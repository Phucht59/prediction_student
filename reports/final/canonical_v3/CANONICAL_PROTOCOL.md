# Canonical V3 protocol

Authority: `UNIFIED_CANONICAL_BENCHMARK_V3`

This is a canonical nested-CV benchmark, not a never-seen external holdout.
UCI uses one frozen `UCICNNBiLSTM` topology across MAT/POR and stage/main
checkpoints. OULAD uses one frozen H1 topology with shared-stage and dedicated
FINAL checkpoints. Outer labels never select configurations or thresholds.

Information policy: **configs/canonical_v3/oulad_information_policy.yaml**.
