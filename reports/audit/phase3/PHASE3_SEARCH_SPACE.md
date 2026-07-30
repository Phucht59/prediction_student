# Phase 3 — Search Space

| Dimension | Space |
| --- | --- |
| Learning rate | 1e-4 to 2e-3, log |
| Weight decay | 1e-8 to 5e-4, log |
| Dropout | 0.10 to 0.35 |
| Batch size | 128 or 256 |
| Loss | standard BCE or weighted BCE |
| Positive weight | sqrt-ratio or full-ratio, inner-train only |
| Survival weight | 0, 0.10, 0.15, 0.20 |
| Outcome weight | 0, 0.10, 0.15, 0.20 |

Optimizer (AdamW), scheduler (none), branch dropout, all architecture
dimensions, pooling, fusion, pretraining, epoch cap, and threshold semantics
were frozen.
