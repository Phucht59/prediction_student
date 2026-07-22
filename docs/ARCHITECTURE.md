# Architecture

The public system has three dataset-specific CNN-BiLSTM facades backed by immutable selected checkpoints. Frozen probabilities feed classification, calibration and ranking audits. OULAD risk profiles then feed the Student Risk-Based Recommendation System, whose safeguards check conflicts, duplicates, workload and lineage before expert review.

The release layer is read-only: `src/final_release` aggregates existing evidence into canonical JSON/CSV and Markdown reports. It has no training entry point.
