from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


RAW_DIR = Path("data/raw")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download optional raw education datasets.")
    parser.add_argument(
        "--uci-student-performance",
        action="store_true",
        help="Download the UCI Student Performance bundle into data/raw/.",
    )
    return parser.parse_args()


def download_uci_student_performance() -> None:
    """Download the UCI Student Performance dataset for local raw storage."""
    from ucimlrepo import fetch_ucirepo

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    dataset = fetch_ucirepo(id=320)
    data = pd.concat([dataset.data.features, dataset.data.targets], axis=1)
    output_path = RAW_DIR / "student_performance_uci.csv"
    data.to_csv(output_path, index=False)
    print(f"Saved UCI Student Performance data to {output_path}")
    print("Main experiments expect student-mat.csv and student-por.csv in data/raw/.")


def main() -> None:
    args = parse_args()
    if args.uci_student_performance:
        download_uci_student_performance()
    else:
        print("No dataset selected. Use --uci-student-performance to download the UCI bundle.")


if __name__ == "__main__":
    main()
