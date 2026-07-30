"""Print the compact Phase 3 status JSON without reading verbose logs."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.training.phase3_optuna import status  # noqa: E402


if __name__ == "__main__":
    print(json.dumps(status(), sort_keys=True))
