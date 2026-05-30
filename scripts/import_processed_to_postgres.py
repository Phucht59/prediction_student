from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.storage.postgres import create_postgres_engine, import_student_records  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Import processed raw split CSV files into PostgreSQL.")
    parser.add_argument("--dataset", required=True, help="Dataset name, for example student-mat or xapi.")
    parser.add_argument("--scenario", required=True, help="Scenario name, for example late or xapi_behavior.")
    parser.add_argument(
        "--processed-dir",
        type=Path,
        default=None,
        help="Directory containing train_raw.csv, val_raw.csv, and test_raw.csv.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Show row counts without writing to PostgreSQL.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    processed_dir = args.processed_dir or PROJECT_ROOT / "data" / "processed" / args.dataset / args.scenario
    split_paths = {
        "train": processed_dir / "train_raw.csv",
        "validation": processed_dir / "val_raw.csv",
        "test": processed_dir / "test_raw.csv",
    }
    missing = [str(path) for path in split_paths.values() if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing processed split files: {missing}")

    split_frames = {split: pd.read_csv(path) for split, path in split_paths.items()}
    for split, frame in split_frames.items():
        print(f"{split}: {len(frame)} rows")

    if args.dry_run:
        print("Dry run complete. No database rows were written.")
        return 0

    engine = create_postgres_engine()
    total_inserted = 0
    for split, frame in split_frames.items():
        inserted = import_student_records(
            engine,
            frame,
            dataset_name=args.dataset,
            scenario=args.scenario,
            split_name=split,
        )
        total_inserted += inserted
        print(f"{split}: inserted {inserted} new rows")

    print(f"Import complete. Inserted {total_inserted} new rows.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
