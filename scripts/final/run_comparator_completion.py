#!/usr/bin/env python
"""Run the preregistered final comparator completion with fold-level resume."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.final_release.comparator_completion import run_training


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        choices=("student_mat", "student_por", "oulad", "all"),
        default="all",
    )
    args = parser.parse_args()
    run_training(args.dataset)


if __name__ == "__main__":
    main()
