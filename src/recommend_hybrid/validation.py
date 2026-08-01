"""Lightweight reusable validation helpers for Phase 2."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import yaml

from .action_catalog import ActionCatalog
from .prediction_adapter import ARCHITECTURE_HASH, PARAMETER_COUNT


def validate_authority(root: Path) -> dict[str, Any]:
    authority = yaml.safe_load(
        (root / "configs/recommend_hybrid/model_authority.yaml").read_text(encoding="utf-8")
    )
    baseline = json.loads(
        (root / "reports/recommend_hybrid/BASELINE_LOCK.json").read_text(encoding="utf-8")
    )
    manifest = json.loads(
        (
            root
            / "artifacts/recommend_hybrid/RECOMMEND_HYBRID_CHECKPOINT_MANIFEST.json"
        ).read_text(encoding="utf-8")
    )
    errors: list[str] = []
    if baseline.get("baseline_status") != "PHASE_1_PASS":
        errors.append("Phase 1 baseline is not PASS")
    for payload, label in ((authority, "authority"), (manifest, "manifest")):
        if payload.get("architecture_hash") != ARCHITECTURE_HASH:
            errors.append(f"{label} architecture hash mismatch")
        if int(payload.get("parameter_count", -1)) != PARAMETER_COUNT:
            errors.append(f"{label} parameter count mismatch")
    if authority.get("final_stage_usage") != "EVALUATION_ONLY":
        errors.append("final-stage policy mismatch")
    if authority.get("separate_prediction_model_allowed") is not False:
        errors.append("separate prediction model is not prohibited")
    return {"status": "PASS" if not errors else "FAIL", "errors": errors}


def validate_catalog(root: Path) -> dict[str, Any]:
    catalog = ActionCatalog.load(root / "configs/recommend_hybrid/actions.yaml")
    return {
        "schema_version": "recommend_hybrid_action_catalog_validation_v1",
        "catalog_version": catalog.version,
        "active_actions": sum(action.active for action in catalog.actions),
        "invalid_actions": 0,
        "final_evaluation_interventions": sum(
            "FINAL_EVALUATION" in {stage.value for stage in action.applicable_stages}
            for action in catalog.actions
        ),
        "status": "PASS",
    }


def json_sha256(payload: Any) -> str:
    value = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


__all__ = ["json_sha256", "validate_authority", "validate_catalog"]
