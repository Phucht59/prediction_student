"""C0 PredictionResult → V3 recommendation fields. No H1, no seed_disagreement."""

from __future__ import annotations

import math

from src.prediction.contracts import PredictionResult

from .contracts import STAGE_FRACTION, map_prediction_state


def binary_entropy(probability: float) -> float:
    p = min(max(float(probability), 1e-12), 1.0 - 1e-12)
    return float(-(p * math.log(p) + (1.0 - p) * math.log(1.0 - p)) / math.log(2.0))


def prediction_result_to_v3_fields(result: PredictionResult) -> dict:
    if result.model_id != "hybrid":
        raise ValueError("V3 accepts only model_id='hybrid'")
    stage = map_prediction_state(result.stage_or_endpoint)
    uncertainty = (
        float(result.uncertainty)
        if result.uncertainty is not None
        else binary_entropy(result.risk_probability)
    )
    payload = {
        "record_id": result.record_id,
        "student_key": result.record_id,
        "stage": stage,
        "risk_probability": float(result.risk_probability),
        "predicted_risk": int(result.predicted_risk),
        "prediction_threshold": float(result.threshold),
        "uncertainty": uncertainty,
        "course_progress": STAGE_FRACTION[stage],
        "model_id": result.model_id,
        "dataset": result.dataset,
        "stage_or_endpoint": result.stage_or_endpoint,
    }
    meta = result.metadata or {}
    if "student_key" in meta:
        payload["student_key"] = str(meta["student_key"])
    if "course_key" in meta:
        payload["course_key"] = str(meta["course_key"])
    if "cutoff_day" in meta:
        payload["cutoff_day"] = int(meta["cutoff_day"])
    return payload


__all__ = ["binary_entropy", "prediction_result_to_v3_fields"]
