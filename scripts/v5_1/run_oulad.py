from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.studies.v5_1.oulad.runner import run_oulad_v5_1  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the frozen OULAD V5.1 protocol")
    parser.add_argument(
        "--phase",
        choices=["architecture", "screen", "search", "final", "all"],
        default="all",
    )
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    print(json.dumps(run_oulad_v5_1(phase=args.phase, device=args.device, force=args.force), indent=2, default=str))


if __name__ == "__main__":
    main()
