from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.studies.common.hashing import sha256_file
from src.studies.oulad.materialize import materialize_all, rebuild_derived_from_sequences


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, default=ROOT / "configs" / "extension_protocol_v1.yaml")
    parser.add_argument("--output", type=Path, default=ROOT / "data" / "processed" / "study_c_oulad")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--rebuild-derived", action="store_true")
    args = parser.parse_args()
    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    for source_id, source in protocol["sources"].items():
        if source_id.startswith("oulad") and sha256_file(ROOT / source["path"]) != source["sha256"]:
            raise RuntimeError(f"Frozen source hash mismatch: {source_id}")
    completion = args.output / "manifests" / "materialization_complete.json"
    if args.rebuild_derived:
        result = rebuild_derived_from_sequences(args.output)
        print(json.dumps(result, indent=2))
        return 0
    if args.resume and completion.exists():
        existing = json.loads(completion.read_text(encoding="utf-8"))
        if existing.get("status") == "PASS":
            print(json.dumps({"status": "SKIPPED_ALREADY_PASS", "output": str(args.output)}, indent=2))
            return 0
    result = materialize_all(ROOT / "data" / "raw", args.output, protocol)
    completion.parent.mkdir(parents=True, exist_ok=True)
    completion.write_text(json.dumps({**result, "completed_at": datetime.now(timezone.utc).isoformat()}, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": result["status"], "forecasts": {key: value["row_count"] for key, value in result["manifests"].items()}, "output": str(args.output)}, indent=2))
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
