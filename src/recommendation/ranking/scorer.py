"""Score one Student State case with the five frozen EBM models."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from ..feasibility.rules_v2 import evaluate_feasibility_v2
from ..models.ebm import load_model, predict_clipped, top_local_reasons
from ..models.features import ACTION_KEYS, ACTION_TO_KEY, APPROVED_FEATURES, encode_state_features
from ..weak_supervision.matrix import FINAL_ACTIONS

ACTION_ORDER = FINAL_ACTIONS
ACTION_QUALITY = {
    "assessment_recovery": "PASS",
    "re_engagement": "PASS",
    "study_planning": "PASS",
    "progress_monitoring": "PASS_WITH_WARNING",
    "retrieval_practice": "REVIEW",
}


def load_ebm_bundle(manifest: dict, root: Path) -> dict:
    models = {}
    for action_id, item in manifest["models"].items():
        models[action_id] = {
            "model": load_model(root / item["artifact_path"]),
            "version": item.get("label_model_version") or manifest.get("version"),
            "quality_status": item["quality_status"],
        }
    if set(models) != set(FINAL_ACTIONS):
        raise ValueError("EBM bundle must contain exactly five action models")
    return models


def score_case(state_row, models: dict, *, model_version: str) -> list[dict]:
    features = encode_state_features(state_row.to_frame().T if hasattr(state_row, "to_frame") else state_row)
    x = features.to_numpy(dtype=float)[0]
    values = state_row.to_dict() if hasattr(state_row, "to_dict") else dict(state_row)
    rows = []
    for action_id in ACTION_ORDER:
        raw, clipped = predict_clipped(models[action_id]["model"], x.reshape(1, -1))
        feasibility_status, reason, source = evaluate_feasibility_v2(values, ACTION_TO_KEY[action_id])
        local = top_local_reasons(models[action_id]["model"], x)
        rows.append({
            "case_id": str(values["case_id"]),
            "action_id": action_id,
            "raw_score": float(raw[0]),
            "relevance_score": float(clipped[0]),
            "feasibility_status": feasibility_status,
            "feasibility_reason": reason,
            "feasibility_source": source,
            "quality_warning": ACTION_QUALITY[action_id],
            "top_positive_reasons": local["top_positive_reasons"],
            "top_negative_reasons": local["top_negative_reasons"],
            "intercept": local["intercept"],
            "model_version": model_version,
            "feature_vector": {name: float(value) for name, value in zip(APPROVED_FEATURES, x)},
        })
    return rows
