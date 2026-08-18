"""Validate Phase 7 silver-label artifact."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.recommendation.weak_supervision.matrix import panel_case_ids  # noqa: E402
from src.recommendation.weak_supervision.silver import validate_silver  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--silver", type=Path, default=ROOT / "artifacts/recommendation/weak_supervision/silver_labels.parquet")
    parser.add_argument("--panel-a", type=Path, default=ROOT / "artifacts/recommendation/panels/panel_a.parquet")
    parser.add_argument("--panel-b", type=Path, default=ROOT / "artifacts/recommendation/panels/panel_b.parquet")
    args = parser.parse_args()
    frame = pd.read_parquet(args.silver)
    _, panel_a = panel_case_ids(args.panel_a)
    _, panel_b = panel_case_ids(args.panel_b)
    validate_silver(frame, panel_a, panel_b)
    print(json.dumps({
        "status": "PASS",
        "rows": int(len(frame)),
        "silver_status": frame["silver_status"].value_counts().astype(int).to_dict(),
        "panel_b_overlap": 0,
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
