from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
import yaml

from src.recommend_hybrid.hybrid_only_final.runtime import (
    load_released_hybrid_only_config,
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_runtime_fails_closed_without_release(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError, match="not released"):
        load_released_hybrid_only_config(tmp_path)


def test_runtime_rejects_nonvalidated_release(tmp_path: Path) -> None:
    config_dir = tmp_path / "configs/recommend_hybrid"
    artifact_dir = tmp_path / "artifacts/recommend_hybrid/hybrid_only_final"
    config_dir.mkdir(parents=True)
    artifact_dir.mkdir(parents=True)
    release = artifact_dir / "HYBRID_ONLY_RELEASE.json"
    release.write_text(
        json.dumps(
            {
                "status": "HYBRID_ONLY_SILVER_EVIDENCE_BELOW_GATE",
                "runtime_authorized": False,
            }
        ),
        encoding="utf-8",
    )
    (config_dir / "hybrid_only_selected.yaml").write_text(
        yaml.safe_dump(
            {
                "status": "RELEASED",
                "release_status": "HYBRID_ONLY_OFFLINE_SILVER_VALIDATED",
                "learned_model": "frozen_residual_cnn_bilstm",
                "additional_learned_ranker": False,
                "release_artifact": str(release.relative_to(tmp_path)).replace("\\", "/"),
                "release_sha256": sha256(release),
                "config": {"config_id": "x"},
                "normalization_scales": {},
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(RuntimeError, match="does not authorize runtime"):
        load_released_hybrid_only_config(tmp_path)


def test_runtime_loads_only_validated_deterministic_config(tmp_path: Path) -> None:
    config_dir = tmp_path / "configs/recommend_hybrid"
    artifact_dir = tmp_path / "artifacts/recommend_hybrid/hybrid_only_final"
    config_dir.mkdir(parents=True)
    artifact_dir.mkdir(parents=True)
    release = artifact_dir / "HYBRID_ONLY_RELEASE.json"
    release.write_text(
        json.dumps(
            {
                "status": "HYBRID_ONLY_OFFLINE_SILVER_VALIDATED",
                "runtime_authorized": True,
            }
        ),
        encoding="utf-8",
    )
    config = {
        "config_id": "abc",
        "risk_weight": 0.8,
        "evidence_weight": 0.2,
        "need_weight": 0.1,
        "certainty_weight": 0.1,
        "workload_weight": 0.05,
        "minimum_risk_reduction": 0.01,
        "maximum_uncertainty": 0.2,
        "minimum_evidence": 0.4,
        "minimum_top_margin": 0.02,
        "minimum_top_score": 0.15,
    }
    scales = {
        "risk_scale": 0.1,
        "need_scale": 1.0,
        "uncertainty_scale": 0.1,
        "workload_scale_minutes": 150.0,
    }
    (config_dir / "hybrid_only_selected.yaml").write_text(
        yaml.safe_dump(
            {
                "status": "RELEASED",
                "release_status": "HYBRID_ONLY_OFFLINE_SILVER_VALIDATED",
                "learned_model": "frozen_residual_cnn_bilstm",
                "additional_learned_ranker": False,
                "release_artifact": str(release.relative_to(tmp_path)).replace("\\", "/"),
                "release_sha256": sha256(release),
                "config": config,
                "normalization_scales": scales,
            }
        ),
        encoding="utf-8",
    )
    loaded = load_released_hybrid_only_config(tmp_path)
    assert loaded.version == "hybrid_only_final_abc"
    assert loaded.risk_weight == 0.8
