"""Deterministic LF declarations and restricted candidate access."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from src.recommend_hybrid.exceptions import ContractValidationError

from .labels import LF_ABSTAIN, TARGET_VALUES


@dataclass(frozen=True)
class LabelingFunction:
    lf_id: str
    lf_family: str
    supported_datasets: tuple[str, ...]
    supported_stages: tuple[str, ...]
    supported_action_ids: tuple[str, ...]
    required_fields: tuple[str, ...]
    source_ids: tuple[str, ...]
    rationale: str
    version: str
    implementation: Callable[[dict[str, Any]], int]

    def __call__(self, row: dict[str, Any]) -> int:
        allowed = set(self.required_fields) | {"dataset", "stage", "action_id", "action_status", "action_datasets", "action_stages", "human_review_required", "prediction_risk", "uncertainty", "missingness_flags"}
        exposed = {key: row[key] for key in allowed if key in row}
        value = self.implementation(exposed)
        if value not in {*TARGET_VALUES, LF_ABSTAIN}:
            raise ContractValidationError(f"{self.lf_id} returned invalid vote")
        return value


__all__ = ["LabelingFunction"]
