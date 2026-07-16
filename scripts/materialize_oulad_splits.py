from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.studies.common.hashing import sha256_file
from src.studies.oulad.cohort import FORECASTS
from src.studies.oulad.splits import build_common_split_manifests


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, default=ROOT / "configs" / "extension_protocol_v1.yaml")
    parser.add_argument("--processed", type=Path, default=ROOT / "data" / "processed" / "study_c_oulad")
    args = parser.parse_args()
    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    frames = {forecast: (pd.read_parquet(args.processed / "cohorts" / f"{forecast}.parquet"), pd.read_parquet(args.processed / "targets" / f"{forecast}.parquet")) for forecast in FORECASTS}
    manifest, future, audit = build_common_split_manifests(frames, protocol["study_c"]["future_support"], seed=42)
    path = args.processed / "manifests" / "split_manifest.csv"
    future_path = args.processed / "manifests" / "future_test_manifest.csv"
    audit_path = args.processed / "manifests" / "future_eligibility_audit.csv"
    manifest.to_csv(path, index=False); future.to_csv(future_path, index=False); audit.to_csv(audit_path, index=False)
    result = {"status": "PASS", "historical_records": int((manifest["role"] == "historical_development").sum()), "future_records": int((manifest["role"] == "future_candidate").sum()), "excluded_overlap_records": int((manifest["role"] == "excluded_future_student_overlap").sum()), "checksums": {"split_manifest": sha256_file(path), "future_test_manifest": sha256_file(future_path), "future_eligibility_audit": sha256_file(audit_path)}}
    (args.processed / "manifests" / "split_complete.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
