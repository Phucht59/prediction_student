"""Bind controls, ablations, and release artifacts to one completed full-grid model authority."""
from __future__ import annotations

import hashlib
import json
import os
import shutil
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "artifacts/recommend_hybrid/outcome_grounded_v2_1"
FINAL = OUT / "final_oof"
SELECTION = OUT / "model_selection"
FULL_MARKER = OUT / "FULL_REGISTERED_SEARCH.json"


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_registered_trials(marker: dict[str, Any]) -> dict[str, Any]:
    expected = int(marker.get("expected_trials_per_outer_fold") or 0)
    if expected <= 0:
        raise RuntimeError("Full-grid marker has no positive expected trial count")
    summaries = []
    for fold in [0, 1, 2]:
        path = SELECTION / f"fold_{fold}_trials.csv"
        if not path.exists():
            raise RuntimeError(f"Missing registered trial table: {path}")
        trials = pd.read_csv(path)
        if len(trials) != expected:
            raise RuntimeError(
                f"Fold {fold} contains {len(trials)} registered trials; expected {expected}"
            )
        if "status" not in trials.columns:
            raise RuntimeError(f"Fold {fold} trial table has no status column")
        incomplete = trials[trials["status"].astype(str) != "COMPLETE"]
        if len(incomplete):
            raise RuntimeError(
                f"Fold {fold} has {len(incomplete)} non-COMPLETE registered trials"
            )
        summaries.append(
            {
                "outer_fold": fold,
                "registered_trials": int(len(trials)),
                "complete_trials": int(len(trials)),
            }
        )
    return {"expected_trials_per_outer_fold": expected, "folds": summaries}


def current_model_authority() -> dict[str, Any]:
    if not FULL_MARKER.exists():
        raise RuntimeError("FULL_REGISTERED_SEARCH.json is missing")
    marker = json.loads(FULL_MARKER.read_text(encoding="utf-8"))
    if marker.get("status") != "COMPLETE":
        raise RuntimeError("Full registered search is not COMPLETE")
    trial_validation = validate_registered_trials(marker)
    required = [FINAL / "NESTED_OOF_RESULTS.json"] + [
        SELECTION / f"fold_{fold}_selected.json" for fold in [0, 1, 2]
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise RuntimeError(f"Missing full-grid authority artifacts: {missing}")
    files = {
        str(path.relative_to(OUT)).replace("\\", "/"): sha256(path)
        for path in required
    }
    combined = hashlib.sha256(
        json.dumps(files, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return {
        "model_authority_sha256": combined,
        "files": files,
        "full_search_status": marker.get("status"),
        **trial_validation,
    }


def next_archive(prefix: str) -> Path:
    candidate = OUT / prefix
    if not candidate.exists():
        return candidate
    number = 2
    while (OUT / f"{prefix}_{number}").exists():
        number += 1
    return OUT / f"{prefix}_{number}"


def prepare_namespace(
    namespace: Path,
    marker_path: Path,
    archive_prefix: str,
    authority: dict[str, Any],
) -> None:
    current_hash = authority["model_authority_sha256"]
    existing: dict[str, Any] = {}
    if marker_path.exists():
        existing = json.loads(marker_path.read_text(encoding="utf-8"))
    existing_hash = existing.get("model_authority_sha256")
    if existing_hash == current_hash:
        return

    if namespace.exists() or marker_path.exists():
        archive = next_archive(archive_prefix)
        archive.mkdir(parents=True, exist_ok=False)
        if namespace.exists():
            shutil.move(str(namespace), str(archive / namespace.name))
        if marker_path.exists():
            shutil.move(str(marker_path), str(archive / marker_path.name))
        atomic_json(
            archive / "ARCHIVE_REASON.json",
            {
                "reason": "POSTSEARCH_MODEL_AUTHORITY_CHANGED",
                "previous_model_authority_sha256": existing_hash,
                "current_model_authority_sha256": current_hash,
            },
        )

    namespace.mkdir(parents=True, exist_ok=True)
    atomic_json(
        marker_path,
        {
            "status": "RUNNING",
            **authority,
        },
    )


def assert_bound(marker_path: Path, authority: dict[str, Any], label: str) -> None:
    if not marker_path.exists():
        raise RuntimeError(f"{label} marker is missing: {marker_path}")
    payload = json.loads(marker_path.read_text(encoding="utf-8"))
    if payload.get("model_authority_sha256") != authority["model_authority_sha256"]:
        raise RuntimeError(f"{label} is bound to a stale model authority")
