"""Run the bounded Phase 11 infrastructure smoke test."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.benchmark.benchmark import run_smoke


if __name__ == "__main__":
    print(json.dumps(run_smoke(), indent=2, sort_keys=True))
