from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.paper_replication.v18_strict_validation import main


if __name__ == "__main__":
    main()
