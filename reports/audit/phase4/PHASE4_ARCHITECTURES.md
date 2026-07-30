# Phase 4 — Architectures

All candidates share temporal backbone hash
`ddc6800df7975a4689af94eef76daf134e78ffb3452b4ed662a743ae00e97b20`. Unique backbone hash count: **1**.

| Architecture | Total | Temporal | Fusion | Heads | Δ vs A0 (%) | Within ±10% |
| --- | --- | --- | --- | --- | --- | --- |
| A0_SCALAR_GATE | 150202 | 126432 | 386 | 5848 | 0.000000 | True |
| A1_VECTOR_GATE | 155080 | 126432 | 5264 | 5848 | 3.247627 | True |
| A2_CONCAT_MLP | 162600 | 126432 | 12784 | 5848 | 8.254218 | True |
| A3_FILM | 154056 | 126432 | 4240 | 5848 | 2.565878 | True |

A0 preserves the Phase 3 state-dict layout and 150,202 parameters. A1 uses
low-rank feature-wise residual gates. A2 uses a two-transform concat MLP with
64-dimensional output. A3 uses zero-initialized FiLM modulation and begins as
the temporal identity. No attention, CNN-depth, dilation, pooling, or recurrent
change was introduced.
