"""Run or smoke-test the Phase 3 Optuna objective."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.training.phase3_optuna import run_smoke  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true", required=True)
    parser.parse_args()
    print(json.dumps(run_smoke(), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
