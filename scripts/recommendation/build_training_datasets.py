"""Build five Phase 8 training tables from frozen Phase 7 silver labels."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.recommendation.models.datasets import build_all_training_sets  # noqa: E402
from src.recommendation.models.features import audit_course_progress, validate_phase7_authority  # noqa: E402
from src.recommendation.weak_supervision.matrix import FINAL_ACTIONS  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--silver", type=Path, default=ROOT / "artifacts/recommendation/weak_supervision/silver_labels.parquet")
    parser.add_argument("--phase7-manifest", type=Path, default=ROOT / "artifacts/recommendation/weak_supervision/phase7_manifest.json")
    parser.add_argument("--phase6-manifest", type=Path, default=ROOT / "artifacts/recommendation/labeling/phase6_source_manifest.json")
    parser.add_argument("--panel-a", type=Path, default=ROOT / "artifacts/recommendation/panels/panel_a.parquet")
    parser.add_argument("--panel-b", type=Path, default=ROOT / "artifacts/recommendation/panels/panel_b.parquet")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "artifacts/recommendation/models/training")
    args = parser.parse_args()
    validate_phase7_authority(args.phase7_manifest, args.silver, args.phase6_manifest)
    silver = pd.read_parquet(args.silver)
    panel_a = pd.read_parquet(args.panel_a)
    panel_b = set(pd.read_parquet(args.panel_b, columns=["case_id"])["case_id"].astype(str))
    if set(panel_a["case_id"].astype(str)) & panel_b:
        raise ValueError("Panel A/B overlap is not zero")
    audit = audit_course_progress(panel_a)
    if not audit["exclude"]:
        raise ValueError("course_progress audit did not confirm stage redundancy")
    tables = build_all_training_sets(silver, panel_a, panel_b, args.output_dir)
    print(json.dumps({
        "course_progress": audit["status"],
        "rows": {action: int(len(tables[action])) for action in FINAL_ACTIONS},
        "panel_b_overlap": 0,
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
