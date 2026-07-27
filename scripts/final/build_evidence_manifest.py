"""Build the immutable public evidence inventory without touching model outputs."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
FINAL_ROOT = ROOT / "artifacts" / "final"
OUTPUT = FINAL_ROOT / "evidence_manifest.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    files = [
        {
            "path": path.relative_to(ROOT).as_posix(),
            "sha256": sha256(path),
        }
        for path in sorted(FINAL_ROOT.rglob("*"))
        if path.is_file() and path != OUTPUT
    ]
    manifest = {
        "schema_version": "final_evidence_manifest_v1",
        "generated_from_immutable_existing_evidence": True,
        "training_performed": False,
        "future_oulad_accessed": False,
        "files": files,
    }
    with OUTPUT.open("w", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({"status": "PASS", "files": len(files)}))


if __name__ == "__main__":
    main()
