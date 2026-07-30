"""Durable Phase 5 supervisor."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.training.phase5_mlp_gap import run_supervisor  # noqa: E402


if __name__ == "__main__":
    raise SystemExit(run_supervisor())
