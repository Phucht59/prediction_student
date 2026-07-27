# Ablation Summary

`artifacts/final/ablation_evidence` preserves CNN-only, BiLSTM-only,
CNN-BiLSTM, aggregate/static, capacity matching, dilation, serial/parallel,
CNN-skip and temporal-order evidence.

- CNN-only and BiLSTM-only in the final tables: **FINAL COMPARATOR**.
- Capacity, dilation, skip, parallel and temporal-order experiments:
  **CONTROLLED DEVELOPMENT DIAGNOSTIC**.
- Engineered XGBoost and frozen earlier-generation references:
  **CROSS-GENERATION COMPARISON**.

The architecture-diagnosis development gate did not select a replacement
model. Its inner results are not final prediction results.
