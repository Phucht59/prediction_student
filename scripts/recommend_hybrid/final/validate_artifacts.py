"""Validate final namespace artifact hashes without training or evaluation."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
MANIFEST = ROOT / "artifacts/recommend_hybrid/final/FINAL_ARTIFACT_MANIFEST.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    data = json.loads(MANIFEST.read_text(encoding="utf-8"))
    assert data["scientific_values_changed"] is False
    for item in data["artifacts"]:
        final_path = ROOT / item["final_path"]
        assert final_path.exists(), final_path
        actual = sha256(final_path)
        assert actual == item["final_sha256"], final_path
    print("FINAL_ARTIFACT_CHECKSUMS_PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
