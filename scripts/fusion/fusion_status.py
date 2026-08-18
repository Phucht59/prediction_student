"""Print the compact Phase 4 status payload once."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.training.fusion_tuning import status  # noqa: E402

print(json.dumps(status(), indent=2, sort_keys=True))
