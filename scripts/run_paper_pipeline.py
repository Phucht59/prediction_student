from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.paper_replication.pipeline import DATASETS, run_cli  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run paper-style CNN-BiLSTM replication pipeline.")
    parser.add_argument("--stage", choices=["prepare", "baseline", "deep", "report", "all"], default="all")
    parser.add_argument("--dataset", choices=["all", *sorted(DATASETS)], default="all")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument(
        "--grid-profile",
        choices=["compact", "full"],
        default="compact",
        help="compact keeps XGBoost grid runnable; full uses the full PDF-like grid.",
    )
    return parser.parse_args()


def main() -> None:
    run_cli(parse_args())


if __name__ == "__main__":
    main()

