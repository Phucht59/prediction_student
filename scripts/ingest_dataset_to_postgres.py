"""Seed a raw CSV dataset into PostgreSQL source_* tables."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import DATASETS
from src.postgres_data_source import ingest_dataset_csv_to_postgres


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=sorted(DATASETS), required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = ingest_dataset_csv_to_postgres(args.dataset)
    print(
        "ingested "
        f"dataset={args.dataset} "
        f"dataset_version_id={result['dataset_version_id']} "
        f"row_count={result['row_count']} "
        f"source_record_count={result['source_record_count']}"
    )


if __name__ == "__main__":
    main()
