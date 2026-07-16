# OULAD Deep V2 preflight

- Branch created from frozen V1 commit `fccaef8b3e73a375f2a9d1bca2cc5897345242bd`.
- V1 baseline suite: **169 passed, 5 skipped, 0 failed**.
- V1 Python/Torch remains unchanged: Python 3.10.8, Torch 2.12.0 CPU.
- Isolated V2 environment: `.venv-oulad-v2`, Torch 2.7.1+cu118.
- CUDA device: NVIDIA GeForce GTX 1650 4 GB, driver 516.22.
- Deterministic CUDA forward/backward, checkpoint reload and repeated-seed smoke: **PASS**, max difference `0.0`.
- Every CUDA job must set `CUBLAS_WORKSPACE_CONFIG=:4096:8` before importing Torch.
- No V2 model training or future-benchmark access occurred before protocol freeze.
