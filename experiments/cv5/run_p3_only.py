"""Resume prompt 3 only: 5-fold Hybrid vs tabular LR/RF. No git push."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.cv5 import run_cv5


def main() -> int:
    sys.argv = ["run_cv5.py", "--dataset", "uci", "--folds", "0,1,2,3,4"]
    run_cv5.main()
    sys.argv = ["run_cv5.py", "--dataset", "oulad", "--folds", "0,1,2,3,4"]
    run_cv5.main()
    from experiments.run_overnight_three_prompts import final_index, log

    final_index()
    log("P3 Hybrid-advantage 5-fold done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
