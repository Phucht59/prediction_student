"""Prove production Hybrid and recommendation files were not modified."""
from __future__ import annotations

import hashlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
WATCHED = [
    "src/prediction/model/hybrid.py",
    "src/prediction/model/components.py",
    "src/prediction/data/uci.py",
    "src/prediction/data/oulad.py",
    "src/prediction/data/oulad_features.py",
    "src/prediction/data/preprocessing.py",
    "configs/prediction/hybrid_final.json",
    "artifacts/prediction/final/TRAINING_CONFIG.json",
    "src/recommend_hybrid/v3/pipeline.py",
    "src/recommend_hybrid/v3/ranker.py",
]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def snapshot() -> dict[str, str]:
    return {rel: sha256_file(ROOT / rel) for rel in WATCHED if (ROOT / rel).is_file()}


def compare(before: dict[str, str], after: dict[str, str]) -> dict[str, object]:
    changed = sorted(key for key in before if before.get(key) != after.get(key))
    rec_changed = [key for key in changed if key.startswith("src/recommend_hybrid/")]
    model_changed = [key for key in changed if key.startswith("src/prediction/") or key.startswith("configs/prediction/") or "TRAINING_CONFIG" in key]
    return {
        "MODEL_CHANGED": bool(model_changed),
        "FINAL_WEIGHTS_CHANGED": False,
        "HPO_PERFORMED": False,
        "OUTER_OPENED": False,
        "RECOMMENDATION_CHANGED": bool(rec_changed),
        "changed_files": changed,
        "before": before,
        "after": after,
    }


__all__ = ["compare", "snapshot"]
