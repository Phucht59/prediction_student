from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.paper_replication.v15_prediction_ensemble import main


if __name__ == "__main__":
    main()
