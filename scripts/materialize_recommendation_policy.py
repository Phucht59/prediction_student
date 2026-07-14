"""Fail-closed retirement entrypoint for the unsafe historical materializer.

Phase D stores governed recommendation snapshots and revisions through its
dedicated schema.  It must not materialize old locked-test predictions.
"""

from __future__ import annotations

import argparse
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Explain why the legacy materializer is unavailable.")
    return parser.parse_args()


def main() -> None:
    parse_args()
    raise SystemExit("Legacy recommendation materialization is disabled; use the governed Phase D pipeline.")


if __name__ == "__main__":
    main()
