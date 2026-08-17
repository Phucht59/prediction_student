from __future__ import annotations

import csv
import hashlib
from pathlib import Path


DEST = Path(__file__).resolve().parents[2]
CURRENT = Path(r"C:\hufit\kltn")
STAGING = Path(r"C:\hufit\kltn_outer_eval_staging_v2")


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def source_for(rel: Path) -> tuple[Path | None, str]:
    s = rel.as_posix()
    if s.startswith("src/hybrid/phase8/"):
        return STAGING / rel, "CODE"
    if s.startswith("src/"):
        return CURRENT / rel, "CODE"
    if s == "scripts/hybrid/run_hybrid_v1.py":
        return CURRENT / rel, "CODE"
    if s.startswith("scripts/"):
        return None, "CODE"
    if s.startswith("tests/"):
        return None, "TEST"
    if s.startswith("data/"):
        return CURRENT / rel, "ARTIFACT"
    if s.startswith("artifacts/hybrid/phase1/"):
        return CURRENT / rel, "ARTIFACT"
    if s.startswith("artifacts/hybrid/phase8/"):
        return STAGING / rel, "ARTIFACT"
    if s.startswith("artifacts/prediction/final/"):
        outer = STAGING / "artifacts/hybrid/phase8/outer_test_final"
        suffix = Path(s).relative_to("artifacts/prediction/final")
        candidate = outer / suffix
        return (candidate if candidate.exists() else None), "ARTIFACT"
    if s.startswith("reports/"):
        return CURRENT / rel, "REPORT"
    if s.startswith("configs/"):
        suffix = Path(s).name
        return STAGING / "artifacts/hybrid/phase8/final_development" / suffix, "CONFIG"
    if s in {"requirements.txt", "pytest.ini", "environment.yml"}:
        return None, "ENVIRONMENT"
    if s in {"README.md", "COPY_MANIFEST.csv"}:
        return None, "ARTIFACT"
    return None, "ARTIFACT"


def main() -> None:
    rows = []
    for path in sorted(DEST.rglob("*")):
        if not path.is_file() or path.name == "COPY_MANIFEST.csv":
            continue
        rel = path.relative_to(DEST)
        if "__pycache__" in rel.parts or path.suffix == ".pyc":
            continue
        source, category = source_for(rel)
        dest_hash = sha(path)
        if source is not None and source.exists():
            source_path = str(source)
            source_hash = sha(source)
            if source_hash != dest_hash:
                raise RuntimeError(f"hash_mismatch:{source}:{path}")
        else:
            source_path = f"generated://{rel.as_posix()}"
            source_hash = dest_hash
        rows.append({"source_path": source_path, "destination_path": str(path), "sha256_source": source_hash, "sha256_destination": dest_hash, "category": category})
    with (DEST / "COPY_MANIFEST.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["source_path", "destination_path", "sha256_source", "sha256_destination", "category"])
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
