"""Small immutable value objects returned by final repositories."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class Dataset:
    dataset_id: int
    slug: str
    display_name: str
    task_type: str
    class_labels: tuple[str, ...]


@dataclass(frozen=True)
class ModelResult:
    dataset: str
    model_key: str
    official_name: str
    is_selected: bool
    metrics: dict[str, float | None]


@dataclass(frozen=True)
class RecommendationPlan:
    plan_id: str
    source_record_id: str
    risk_probability: float
    risk_band: str
    status: str
    payload: dict[str, Any]


__all__ = ["Dataset", "ModelResult", "RecommendationPlan"]
