from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.studies.v5_1.uci import run_uci_v5_1  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the frozen UCI V5.1 protocol")
    parser.add_argument("dataset", choices=["student-mat", "student-por"])
    parser.add_argument("--phase", choices=["screen", "search", "final", "all"], default="all")
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    result = run_uci_v5_1(args.dataset, phase=args.phase, device=args.device, force=args.force)
    print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()
