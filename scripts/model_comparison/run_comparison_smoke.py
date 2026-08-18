"""Run the disposable Phase 5 H1 smoke path."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.training.model_comparison import run_smoke  # noqa: E402

print(json.dumps(run_smoke(), indent=2, sort_keys=True))
