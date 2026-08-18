"""Validate the frozen Phase 10 recommendation authority."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.recommendation.finalization.authority import (  # noqa: E402
    validate_checksums,
    validate_required_artifacts,
    validate_scientific_authority,
)
from src.recommendation.finalization.freeze import write_freeze_artifacts  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--write", action="store_true", help="rewrite freeze + source-of-truth artifacts")
    args = parser.parse_args()
    blockers = validate_required_artifacts(ROOT) + validate_scientific_authority(ROOT)
    freeze_path = ROOT / "artifacts/recommendation/final/FINAL_RECOMMENDATION_FREEZE_MANIFEST.json"
    if args.write or not freeze_path.exists():
        freeze = write_freeze_artifacts(ROOT)
    else:
        freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
    blockers.extend(validate_checksums(ROOT, freeze))
    if blockers:
        print("FINAL_FREEZE_FAIL")
        print(json.dumps(blockers, indent=2))
        return 1
    print("FINAL_FREEZE_PASS")
    print(str(freeze_path))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
