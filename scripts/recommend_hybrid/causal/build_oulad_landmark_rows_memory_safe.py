"""Memory-safe entry point for the authoritative OULAD landmark builder."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from scripts.recommend_hybrid.causal import build_oulad_landmark_rows as base
from src.recommend_hybrid.causal.oulad_activity import (
    collect_weekly_activity_sqlite,
)


def build(
    output_path: Path = base.OUTPUT,
    manifest_path: Path = base.MANIFEST,
    *,
    chunksize: int = 750_000,
    batch_size: int = 512,
    force_bundle: bool = False,
):
    original = base._collect_weekly_activity
    try:
        base._collect_weekly_activity = collect_weekly_activity_sqlite
        return base.build(
            output_path,
            manifest_path,
            chunksize=chunksize,
            batch_size=batch_size,
            force_bundle=force_bundle,
        )
    finally:
        base._collect_weekly_activity = original


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=base.OUTPUT)
    parser.add_argument("--manifest", type=Path, default=base.MANIFEST)
    parser.add_argument("--chunksize", type=int, default=750_000)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--force-bundle", action="store_true")
    args = parser.parse_args()
    frame = build(
        args.output,
        args.manifest,
        chunksize=args.chunksize,
        batch_size=args.batch_size,
        force_bundle=args.force_bundle,
    )
    print(
        json.dumps(
            {
                "status": "COMPLETE",
                "rows": len(frame),
                "output": str(args.output),
                "activity_aggregation": "SQLITE_DISK_BACKED",
            }
        )
    )


if __name__ == "__main__":
    main()
