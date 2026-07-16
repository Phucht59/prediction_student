# OULAD Deep V3 preflight

- Source: exact V2 evidence commit `07217a184b9a5dcc6402e3f117a5af2e84c7596f`.
- Study label: **exploratory post-V2 temporal representation study**.
- Baseline suite: **189 passed, 5 skipped, 0 failed**.
- CUDA environment: PyTorch `2.7.1+cu118`, NVIDIA GeForce GTX 1650.
- Smoke: forward/backward, mixed precision, deterministic initialization, variable lengths, masked attention, finite pooled output and checkpoint replay all PASS.
- CPU-only V1 Python environment was not modified.
