from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.studies.v6_2.validation import validate_final  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run V6.2 recommendation validation without model training"
    )
    parser.add_argument("command", choices=["validate-final"])
    parser.add_argument("--expert-file", type=Path)
    parser.add_argument("--run-tests", action="store_true")
    parser.add_argument(
        "--apply-database-migration",
        action="store_true",
        help="Allowed only with an explicit POSTGRES_TEST_DSN/POSTGRES_TEST_APP_DSN",
    )
    args = parser.parse_args()
    result = validate_final(
        expert_file=args.expert_file,
        run_tests=args.run_tests,
        apply_database_migration=args.apply_database_migration,
    )
    print(
        json.dumps(
            {
                "status": result["status"],
                "expert_status": result["expert"]["status"],
                "database_status": result["database"]["status"],
                "tests": result["tests"]["totals"],
                "future_oulad_accessed": result["future_oulad_accessed"],
                "training_run": result["training_run"],
            },
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
