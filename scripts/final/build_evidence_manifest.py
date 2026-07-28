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
    previous = {}
    if OUTPUT.is_file():
        previous = json.loads(OUTPUT.read_text(encoding="utf-8"))
    files = [
        {
            "path": path.relative_to(ROOT).as_posix(),
            "sha256": sha256(path),
        }
        for path in sorted(FINAL_ROOT.rglob("*"))
        if path.is_file() and path != OUTPUT
    ]
    manifest = {
        "schema_version": previous.get(
            "schema_version", "final_evidence_manifest_v2"
        ),
        "generated_from_immutable_existing_evidence": previous.get(
            "generated_from_immutable_existing_evidence", False
        ),
        "training_performed": previous.get("training_performed", True),
        "official_deep_model_training_performed": previous.get(
            "official_deep_model_training_performed", False
        ),
        "teacher_feedback_comparator_training_performed": previous.get(
            "teacher_feedback_comparator_training_performed", True
        ),
        "outer_evaluation_used_for_selection": previous.get(
            "outer_evaluation_used_for_selection", False
        ),
        "best_seed_selected": previous.get("best_seed_selected", False),
        "future_oulad_accessed": previous.get(
            "future_oulad_accessed", False
        ),
        "files": files,
    }
    with OUTPUT.open("w", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({"status": "PASS", "files": len(files)}))


if __name__ == "__main__":
    main()
