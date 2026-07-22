"""Scientifically frozen V5 study implementations."""

from __future__ import annotations

import os


# CUDA deterministic algorithms require this to be set before the first
# cuBLAS-backed operation.  Package import precedes every V5 Torch module.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
