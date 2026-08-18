"""Run the bounded, resume-safe Phase 9 endpoint recovery study."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.training.endpoint_recovery import run_supervisor  # noqa: E402


if __name__ == "__main__":
    raise SystemExit(run_supervisor())
