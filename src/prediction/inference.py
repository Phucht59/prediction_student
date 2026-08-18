"""Model-neutral prediction output boundary."""

from __future__ import annotations

import math
from typing import Mapping, Sequence

import torch

from .contracts import PredictionResult
from .model import Hybrid


def predict_results(model: Hybrid, inputs: Mapping[str, torch.Tensor], *, dataset: str, record_ids: Sequence[str], stage_or_endpoint: str, threshold: float = 0.5) -> list[PredictionResult]:
    required = {"static", "temporal", "temporal_mask", "lengths", "aggregate", "aggregate_available", "progress"}
    if set(inputs) != required:
        raise ValueError(f"inputs must be exactly {sorted(required)}")
    if not isinstance(model, Hybrid):
        raise TypeError("prediction boundary accepts only Hybrid")
    model.eval()
    with torch.inference_mode():
        logits = model(**inputs)
        probabilities = torch.sigmoid(logits).detach().cpu().reshape(-1).tolist()
    if len(probabilities) != len(record_ids):
        raise ValueError("record_ids and model output length differ")
    uncertainty = [-(p * math.log(max(p, 1e-12)) + (1.0 - p) * math.log(max(1.0 - p, 1e-12))) / math.log(2.0) for p in probabilities]
    return [PredictionResult(dataset=dataset, record_id=str(record_id), stage_or_endpoint=stage_or_endpoint, risk_probability=float(probability), predicted_risk=int(probability >= threshold), threshold=threshold, uncertainty=float(ent), model_id="hybrid") for record_id, probability, ent in zip(record_ids, probabilities, uncertainty)]


__all__ = ["predict_results"]
