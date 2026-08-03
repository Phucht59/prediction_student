"""Configuration-backed action effects for conservative OULAD simulations."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from math import isfinite
from pathlib import Path
from typing import Any, Mapping

import yaml

from src.recommend_hybrid.exceptions import ContractValidationError


class EffectOperator(str, Enum):
    MAX_REFERENCE = "max_reference"
    INCREASE_FRACTION_OF_GAP = "increase_fraction_of_gap"
    REDUCE_FRACTION = "reduce_fraction"
    ADD = "add"
    SUBTRACT = "subtract"
    MAX_VALUE = "max_value"


@dataclass(frozen=True)
class FeatureEffect:
    feature_name: str
    operator: EffectOperator
    amount: float | None = None
    reference_key: str | None = None
    lower_bound: float | None = None
    upper_bound: float | None = None

    def __post_init__(self) -> None:
        if not self.feature_name:
            raise ContractValidationError("feature effect requires a feature name")
        for value in (self.amount, self.lower_bound, self.upper_bound):
            if value is not None and not isfinite(value):
                raise ContractValidationError("effect numeric values must be finite")
        if self.lower_bound is not None and self.upper_bound is not None:
            if self.lower_bound > self.upper_bound:
                raise ContractValidationError("effect lower_bound exceeds upper_bound")
        if self.operator is EffectOperator.MAX_REFERENCE and not self.reference_key:
            raise ContractValidationError("max_reference requires reference_key")
        if (
            self.operator is not EffectOperator.MAX_REFERENCE
            and self.reference_key is not None
        ):
            raise ContractValidationError(
                "reference_key is only valid for max_reference"
            )
        if self.operator in {
            EffectOperator.INCREASE_FRACTION_OF_GAP,
            EffectOperator.REDUCE_FRACTION,
        } and (self.amount is None or not 0.0 < self.amount <= 1.0):
            raise ContractValidationError("fraction operator amount must be in (0, 1]")
        if (
            self.operator is EffectOperator.INCREASE_FRACTION_OF_GAP
            and self.upper_bound is None
        ):
            raise ContractValidationError(
                "increase_fraction_of_gap requires upper_bound"
            )
        if self.operator in {
            EffectOperator.ADD,
            EffectOperator.SUBTRACT,
            EffectOperator.MAX_VALUE,
        } and self.amount is None:
            raise ContractValidationError(f"{self.operator.value} requires amount")


@dataclass(frozen=True)
class ActionEffectSpec:
    action_id: str
    scorable: bool
    effects: tuple[FeatureEffect, ...]
    non_scorable_reason: str | None = None

    def __post_init__(self) -> None:
        if not self.action_id:
            raise ContractValidationError("action effect spec requires action_id")
        features = [item.feature_name for item in self.effects]
        if len(features) != len(set(features)):
            raise ContractValidationError(
                f"duplicate feature effect for {self.action_id}"
            )
        if self.scorable and not self.effects:
            raise ContractValidationError("scorable action requires at least one effect")
        if not self.scorable and self.effects:
            raise ContractValidationError("non-scorable action cannot define effects")
        if not self.scorable and not self.non_scorable_reason:
            raise ContractValidationError("non-scorable action requires a reason")


class CounterfactualEffectCatalog:
    def __init__(
        self,
        *,
        version: str,
        mutable_features: frozenset[str],
        protected_features: frozenset[str],
        actions: tuple[ActionEffectSpec, ...],
    ) -> None:
        self.version = version
        self.mutable_features = mutable_features
        self.protected_features = protected_features
        self.actions = actions
        self.validate()

    @classmethod
    def load(cls, path: Path) -> "CounterfactualEffectCatalog":
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
        actions = tuple(cls._parse_action(row) for row in payload["actions"])
        return cls(
            version=str(payload["version"]),
            mutable_features=frozenset(payload["mutable_features"]),
            protected_features=frozenset(payload["protected_features"]),
            actions=actions,
        )

    @staticmethod
    def _parse_action(row: Mapping[str, Any]) -> ActionEffectSpec:
        effects = tuple(
            FeatureEffect(
                feature_name=str(effect["feature"]),
                operator=EffectOperator(effect["operator"]),
                amount=(
                    float(effect["amount"])
                    if effect.get("amount") is not None
                    else None
                ),
                reference_key=(
                    str(effect["reference_key"])
                    if effect.get("reference_key") is not None
                    else None
                ),
                lower_bound=(
                    float(effect["lower_bound"])
                    if effect.get("lower_bound") is not None
                    else None
                ),
                upper_bound=(
                    float(effect["upper_bound"])
                    if effect.get("upper_bound") is not None
                    else None
                ),
            )
            for effect in row.get("effects", [])
        )
        return ActionEffectSpec(
            action_id=str(row["action_id"]),
            scorable=bool(row["scorable"]),
            effects=effects,
            non_scorable_reason=(
                str(row["non_scorable_reason"])
                if row.get("non_scorable_reason") is not None
                else None
            ),
        )

    def validate(self) -> None:
        if not self.version:
            raise ContractValidationError(
                "counterfactual effect catalog version is required"
            )
        if self.mutable_features & self.protected_features:
            raise ContractValidationError(
                "mutable and protected feature sets overlap"
            )
        ids = [item.action_id for item in self.actions]
        if len(ids) != len(set(ids)):
            raise ContractValidationError("duplicate action effect ID")
        for action in self.actions:
            for effect in action.effects:
                if effect.feature_name not in self.mutable_features:
                    raise ContractValidationError(
                        "effect for "
                        f"{action.action_id} changes non-mutable feature "
                        f"{effect.feature_name}"
                    )
                if effect.feature_name in self.protected_features:
                    raise ContractValidationError(
                        "effect for "
                        f"{action.action_id} changes protected feature "
                        f"{effect.feature_name}"
                    )

    def by_id(self, action_id: str) -> ActionEffectSpec:
        for action in self.actions:
            if action.action_id == action_id:
                return action
        raise KeyError(action_id)


__all__ = [
    "ActionEffectSpec",
    "CounterfactualEffectCatalog",
    "EffectOperator",
    "FeatureEffect",
]
