"""Validate the frozen prediction authority used by the final module."""
from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def main() -> int:
    manifest = json.loads((ROOT / "artifacts/recommend_hybrid/RECOMMEND_HYBRID_CHECKPOINT_MANIFEST.json").read_text())
    assert manifest["status"] == "RELEASE_FROZEN"
    assert manifest["parameter_count"] == 160492
    assert manifest["architecture_hash"]
    assert manifest["missing_checkpoints"] == []
    assert manifest["invalid_checkpoints"] == []
    print("CHECKPOINT_AUTHORITY_PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
