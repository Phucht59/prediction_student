"""Release-gated loader for the final hybrid-only deterministic scorer."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import yaml

from .scorer import HybridOnlyScoreConfig

RELEASE_STATUS = "HYBRID_ONLY_OFFLINE_SILVER_VALIDATED"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load_released_hybrid_only_config(root: Path) -> HybridOnlyScoreConfig:
    """Load the deterministic score only when the fail-closed release passed."""

    config_path = root / "configs/recommend_hybrid/hybrid_only_selected.yaml"
    if not config_path.exists():
        raise RuntimeError("hybrid-only runtime is not released")
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if payload.get("status") != "RELEASED":
        raise RuntimeError("hybrid-only runtime config is not released")
    if payload.get("release_status") != RELEASE_STATUS:
        raise RuntimeError("hybrid-only release status is not authorized")
    if payload.get("learned_model") != "frozen_residual_cnn_bilstm":
        raise RuntimeError("unexpected learned model authority")
    if payload.get("additional_learned_ranker") is not False:
        raise RuntimeError("additional learned recommendation ranker is forbidden")

    release_path = root / str(payload["release_artifact"])
    if not release_path.exists():
        raise RuntimeError("hybrid-only release artifact is missing")
    if _sha256(release_path) != str(payload["release_sha256"]):
        raise RuntimeError("hybrid-only release checksum mismatch")
    release = json.loads(release_path.read_text(encoding="utf-8"))
    if release.get("status") != RELEASE_STATUS or not release.get("runtime_authorized"):
        raise RuntimeError("hybrid-only release artifact does not authorize runtime")

    values: dict[str, Any] = {
        **dict(payload["config"]),
        **dict(payload["normalization_scales"]),
    }
    return HybridOnlyScoreConfig(
        version=f"hybrid_only_final_{payload['config']['config_id']}",
        risk_weight=float(values["risk_weight"]),
        evidence_weight=float(values["evidence_weight"]),
        need_weight=float(values["need_weight"]),
        certainty_weight=float(values["certainty_weight"]),
        workload_weight=float(values["workload_weight"]),
        minimum_risk_reduction=float(values["minimum_risk_reduction"]),
        maximum_uncertainty=float(values["maximum_uncertainty"]),
        minimum_evidence=float(values["minimum_evidence"]),
        minimum_top_margin=float(values["minimum_top_margin"]),
        minimum_top_score=float(values["minimum_top_score"]),
        risk_scale=float(values["risk_scale"]),
        need_scale=float(values["need_scale"]),
        uncertainty_scale=float(values["uncertainty_scale"]),
        workload_scale_minutes=float(values["workload_scale_minutes"]),
    )


__all__ = ["RELEASE_STATUS", "load_released_hybrid_only_config"]
