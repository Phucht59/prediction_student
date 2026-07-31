"""Print only the structured Phase 11 runtime status."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.canonical_v3.benchmark import status


if __name__ == "__main__":
    print(json.dumps(status(), indent=2, sort_keys=True))
