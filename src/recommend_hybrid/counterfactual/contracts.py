"""Validated contracts for model-based counterfactual action ranking."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, fields, is_dataclass
from enum import Enum
from math import isfinite
from typing import Any

from src.recommend_hybrid.exceptions import ContractValidationError

StateValue = float | int | None
StateItems = tuple[tuple[str, StateValue], ...]


class SimulationStatus(str, Enum):
    SIMULATED = "SIMULATED"
    NOT_SCORABLE = "NOT_SCORABLE"
    MISSING_INPUT = "MISSING_INPUT"
    NO_CHANGE = "NO_CHANGE"


class UtilityStatus(str, Enum):
    RANKED = "RANKED"
    REJECTED = "REJECTED"


def _json_value(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if is_dataclass(value):
        return {
            field.name: _json_value(getattr(value, field.name)) for field in fields(value)
        }
    if isinstance(value, tuple | list):
        return [_json_value(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    return value


class CounterfactualContract:
    def to_dict(self) -> dict[str, Any]:
        return _json_value(self)


def _validate_probability(value: float, field_name: str) -> None:
    if not isfinite(value) or not 0.0 <= value <= 1.0:
        raise ContractValidationError(f"{field_name} must be finite in [0, 1]")


def canonical_state_items(state: Mapping[str, StateValue]) -> StateItems:
    items = tuple(sorted((str(key), value) for key, value in dict(state).items()))
    for key, value in items:
        if not key:
            raise ContractValidationError("state feature name cannot be empty")
        if value is not None and (
            isinstance(value, bool) or not isinstance(value, int | float)
        ):
            raise ContractValidationError(f"state feature {key} must be numeric or None")
        if isinstance(value, int | float) and not isfinite(float(value)):
            raise ContractValidationError(f"state feature {key} must be finite")
    return items


@dataclass(frozen=True)
class FeatureChange(CounterfactualContract):
    feature_name: str
    before: StateValue
    after: StateValue
    operation: str
    reference_key: str | None = None

    def __post_init__(self) -> None:
        if not self.feature_name or not self.operation:
            raise ContractValidationError("feature change identity is required")
        for value in (self.before, self.after):
            if value is not None and (
                isinstance(value, bool) or not isinstance(value, int | float)
            ):
                raise ContractValidationError(
                    "feature change values must be numeric or None"
                )
            if isinstance(value, int | float) and not isfinite(float(value)):
                raise ContractValidationError("feature change values must be finite")
        if self.before == self.after:
            raise ContractValidationError("feature change must alter the value")


@dataclass(frozen=True)
class CounterfactualScenario(CounterfactualContract):
    action_id: str
    baseline_state: StateItems
    simulated_state: StateItems
    changes: tuple[FeatureChange, ...]
    status: SimulationStatus
    reason_codes: tuple[str, ...]
    simulator_version: str

    def __post_init__(self) -> None:
        if not self.action_id or not self.simulator_version:
            raise ContractValidationError("scenario identity and version are required")
        if not self.reason_codes:
            raise ContractValidationError("scenario requires reason codes")
        if self.status is SimulationStatus.SIMULATED and not self.changes:
            raise ContractValidationError(
                "simulated scenario requires at least one feature change"
            )
        if self.status is not SimulationStatus.SIMULATED and self.changes:
            raise ContractValidationError("non-simulated scenario cannot expose changes")
        if tuple(sorted(self.baseline_state)) != self.baseline_state:
            raise ContractValidationError("baseline state must be canonical")
        if tuple(sorted(self.simulated_state)) != self.simulated_state:
            raise ContractValidationError("simulated state must be canonical")

    def baseline_mapping(self) -> dict[str, StateValue]:
        return dict(self.baseline_state)

    def simulated_mapping(self) -> dict[str, StateValue]:
        return dict(self.simulated_state)


@dataclass(frozen=True)
class RiskEstimate(CounterfactualContract):
    risk_probability: float
    uncertainty: float
    source: str

    def __post_init__(self) -> None:
        _validate_probability(self.risk_probability, "risk_probability")
        _validate_probability(self.uncertainty, "uncertainty")
        if not self.source:
            raise ContractValidationError("risk estimate source is required")


@dataclass(frozen=True)
class ActionUtility(CounterfactualContract):
    action_id: str
    status: UtilityStatus
    baseline_risk: float
    counterfactual_risk: float
    risk_reduction: float
    evidence_strength: float
    uncertainty_penalty: float
    workload_minutes: int
    utility_score: float
    changes: tuple[FeatureChange, ...]
    reason_codes: tuple[str, ...]

    def __post_init__(self) -> None:
        if not self.action_id or not self.reason_codes:
            raise ContractValidationError("action utility identity and reasons are required")
        _validate_probability(self.baseline_risk, "baseline_risk")
        _validate_probability(self.counterfactual_risk, "counterfactual_risk")
        _validate_probability(self.evidence_strength, "evidence_strength")
        _validate_probability(self.uncertainty_penalty, "uncertainty_penalty")
        expected_reduction = self.baseline_risk - self.counterfactual_risk
        if abs(self.risk_reduction - expected_reduction) > 1e-8:
            raise ContractValidationError("risk_reduction is inconsistent")
        if not isfinite(self.utility_score) or self.utility_score < 0.0:
            raise ContractValidationError("utility_score must be finite and non-negative")
        if not 0 < self.workload_minutes <= 180:
            raise ContractValidationError("workload_minutes must be in [1, 180]")
        if self.status is UtilityStatus.RANKED and (
            self.risk_reduction <= 0.0 or not self.changes
        ):
            raise ContractValidationError(
                "ranked action requires positive reduction and changes"
            )


@dataclass(frozen=True)
class CounterfactualRankingResult(CounterfactualContract):
    baseline_estimate: RiskEstimate
    ranked_actions: tuple[ActionUtility, ...]
    rejected_actions: tuple[ActionUtility, ...]
    ranker_version: str

    def __post_init__(self) -> None:
        if not self.ranker_version:
            raise ContractValidationError("ranker version is required")
        all_actions = (*self.ranked_actions, *self.rejected_actions)
        ids = [item.action_id for item in all_actions]
        if len(ids) != len(set(ids)):
            raise ContractValidationError("ranking result contains duplicate actions")
        if any(
            item.status is not UtilityStatus.RANKED for item in self.ranked_actions
        ):
            raise ContractValidationError("ranked_actions contains a rejected item")
        if any(
            item.status is not UtilityStatus.REJECTED for item in self.rejected_actions
        ):
            raise ContractValidationError("rejected_actions contains a ranked item")


__all__ = [
    "ActionUtility",
    "CounterfactualContract",
    "CounterfactualRankingResult",
    "CounterfactualScenario",
    "FeatureChange",
    "RiskEstimate",
    "SimulationStatus",
    "StateItems",
    "StateValue",
    "UtilityStatus",
    "canonical_state_items",
]
