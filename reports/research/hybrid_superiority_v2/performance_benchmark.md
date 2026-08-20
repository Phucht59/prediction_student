# Performance benchmark

| Setting | Throughput / note |
|---|---|
| FP32 GEMM 2048 | 4.52 TFLOPS |
| FP16 AMP GEMM 2048 | 19.34 TFLOPS (selected) |
| BF16 AMP GEMM 2048 | 2.74 TFLOPS (not selected) |
| TF32 | unsupported on CC 7.5 |
| Hybrid AMP | GradScaler FP16 |
| Compile | probe later; keep only if ≥5% and numerically stable |
| CPU baselines | n_jobs/thread_count=4 |

Numerical parity: confirmation mode disables cudnn.benchmark. HPO may enable it.
