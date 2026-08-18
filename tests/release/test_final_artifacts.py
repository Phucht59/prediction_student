from __future__ import annotations

import hashlib
import json
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[2]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def test_all_final_checkpoints_load_and_match_checksums() -> None:
    manifest = json.loads(
        (
            ROOT / "artifacts/final/checksums/checkpoint_manifest.json"
        ).read_text(encoding="utf-8-sig")
    )
    assert manifest["checkpoint_count"] == 65
    for entry in manifest["checkpoints"]:
        path = ROOT / entry["path"]
        assert _sha256(path) == entry["sha256"]
        state = torch.load(path, map_location="cpu", weights_only=True)
        assert isinstance(state, dict) and state


def test_canonical_headline_metrics_are_frozen() -> None:
    payload = json.loads(
        (ROOT / "artifacts/final/final_results.json").read_text(encoding="utf-8")
    )
    expected = {
        "student_mat": (0.9014601961315334, 0.9020888215665611, 0.9441838635944574),
        "student_por": (0.8622587167738002, 0.8675763663148155, 0.914678708867879),
        "oulad": (0.8280835945631038, 0.8203325597306252, 0.893354764945997),
    }
    for dataset, values in expected.items():
        row = payload["datasets"][dataset]["models"][0]
        assert (
            row["metrics"]["macro_f1"]["value"],
            row["metrics"]["balanced_accuracy"]["value"],
            row["metrics"]["pr_auc"]["value"],
        ) == values


def test_corrected_recommendation_semantics() -> None:
    technical = json.loads(
        (
            ROOT
            / "artifacts/final/recommendation/recommendation_technical_validation.json"
        ).read_text(encoding="utf-8")
    )
    assert technical["records"] == 15378
    assert technical["status_counts"] == {
        "ABSTAINED": 3216,
        "GENERATED": 10953,
        "PARTIAL_EVIDENCE": 1209,
    }
    assert technical["deterministic_replay"] is True
    assert technical["future_oulad_accessed"] is False
    assert technical["post_cutoff_used"] is False
