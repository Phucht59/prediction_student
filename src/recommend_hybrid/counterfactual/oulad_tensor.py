"""Counterfactual OULAD tensor simulation against the frozen prediction authority."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Protocol

import numpy as np
import torch
import yaml

from src.recommend_hybrid.exceptions import ContractValidationError

from .contracts import (
    CounterfactualRankingResult,
    CounterfactualScenario,
    FeatureChange,
    RiskEstimate,
    SimulationStatus,
    canonical_state_items,
)
from .ranker import CounterfactualUtilityConfig, CounterfactualUtilityRanker

BASE_CHANNELS = (
    "total_clicks",
    "active_days",
    "unique_sites",
    "unique_activity_types",
    "content_clicks",
    "forum_clicks",
    "quiz_clicks",
    "assessment_related_clicks",
    "submitted_assessment_count",
    "late_submission_count",
    "available_score_count",
    "cumulative_mean_score",
    "cumulative_weighted_score",
    "days_since_last_vle_activity",
    "weeks_without_activity",
    "score_missing_mask",
)
SCORE_CHANNELS = frozenset(
    {
        "available_score_count",
        "cumulative_mean_score",
        "cumulative_weighted_score",
        "score_missing_mask",
    }
)
MODEL_INPUT_KEYS = frozenset(
    {"sequence", "lengths", "mask", "aggregate", "static"}
)


class TensorEffectOperator(str, Enum):
    MAX_REFERENCE = "max_reference"
    ADD = "add"


@dataclass(frozen=True)
class TensorChannelEffect:
    channel_name: str
    operator: TensorEffectOperator
    reference_key: str | None = None
    amount: float | None = None

    def __post_init__(self) -> None:
        if not self.channel_name:
            raise ContractValidationError("tensor effect requires channel_name")
        if self.operator is TensorEffectOperator.MAX_REFERENCE:
            if not self.reference_key or self.amount is not None:
                raise ContractValidationError(
                    "max_reference requires only reference_key"
                )
        if self.operator is TensorEffectOperator.ADD:
            if self.amount is None or self.reference_key is not None:
                raise ContractValidationError("add requires only amount")
            if self.amount <= 0.0:
                raise ContractValidationError("add amount must be positive")


@dataclass(frozen=True)
class TensorActionEffectSpec:
    action_id: str
    scorable: bool
    lookback_weeks: int
    effects: tuple[TensorChannelEffect, ...]
    non_scorable_reason: str | None = None

    def __post_init__(self) -> None:
        if not self.action_id:
            raise ContractValidationError("tensor action requires action_id")
        if self.scorable and (self.lookback_weeks <= 0 or not self.effects):
            raise ContractValidationError(
                "scorable tensor action requires lookback and effects"
            )
        if not self.scorable and (
            self.lookback_weeks != 0
            or self.effects
            or not self.non_scorable_reason
        ):
            raise ContractValidationError(
                "non-scorable tensor action requires only a fallback reason"
            )
        channels = [effect.channel_name for effect in self.effects]
        if len(channels) != len(set(channels)):
            raise ContractValidationError(
                f"duplicate tensor channel for {self.action_id}"
            )


class OULADTensorEffectCatalog:
    def __init__(
        self,
        *,
        version: str,
        mutable_channels: frozenset[str],
        protected_channels: frozenset[str],
        actions: tuple[TensorActionEffectSpec, ...],
        utility_config: CounterfactualUtilityConfig,
    ) -> None:
        self.version = version
        self.mutable_channels = mutable_channels
        self.protected_channels = protected_channels
        self.actions = actions
        self.utility_config = utility_config
        self.validate()

    @classmethod
    def load(cls, path: Path) -> "OULADTensorEffectCatalog":
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
        actions = tuple(cls._parse_action(row) for row in payload["actions"])
        utility = payload["utility"]
        return cls(
            version=str(payload["version"]),
            mutable_channels=frozenset(payload["mutable_channels"]),
            protected_channels=frozenset(payload["protected_channels"]),
            actions=actions,
            utility_config=CounterfactualUtilityConfig(
                version=str(utility["version"]),
                minimum_risk_reduction=float(
                    utility["minimum_risk_reduction"]
                ),
                workload_scale_minutes=float(
                    utility["workload_scale_minutes"]
                ),
            ),
        )

    @staticmethod
    def _parse_action(row: Mapping[str, Any]) -> TensorActionEffectSpec:
        effects = tuple(
            TensorChannelEffect(
                channel_name=str(effect["channel"]),
                operator=TensorEffectOperator(effect["operator"]),
                reference_key=(
                    str(effect["reference_key"])
                    if effect.get("reference_key") is not None
                    else None
                ),
                amount=(
                    float(effect["amount"])
                    if effect.get("amount") is not None
                    else None
                ),
            )
            for effect in row.get("effects", [])
        )
        return TensorActionEffectSpec(
            action_id=str(row["action_id"]),
            scorable=bool(row["scorable"]),
            lookback_weeks=int(row.get("lookback_weeks", 0)),
            effects=effects,
            non_scorable_reason=(
                str(row["non_scorable_reason"])
                if row.get("non_scorable_reason") is not None
                else None
            ),
        )

    def validate(self) -> None:
        if not self.version:
            raise ContractValidationError("tensor effect version is required")
        known = set(BASE_CHANNELS)
        if not self.mutable_channels <= known:
            raise ContractValidationError("unknown mutable OULAD channel")
        if not self.protected_channels <= known:
            raise ContractValidationError("unknown protected OULAD channel")
        if self.mutable_channels & self.protected_channels:
            raise ContractValidationError(
                "mutable and protected tensor channels overlap"
            )
        if not SCORE_CHANNELS <= self.protected_channels:
            raise ContractValidationError(
                "all score channels must remain protected"
            )
        ids = [action.action_id for action in self.actions]
        if len(ids) != len(set(ids)):
            raise ContractValidationError("duplicate tensor action ID")
        for action in self.actions:
            for effect in action.effects:
                if effect.channel_name not in self.mutable_channels:
                    raise ContractValidationError(
                        f"{action.action_id} changes non-mutable channel "
                        f"{effect.channel_name}"
                    )

    def by_id(self, action_id: str) -> TensorActionEffectSpec:
        for action in self.actions:
            if action.action_id == action_id:
                return action
        raise KeyError(action_id)


class OULADFeatureAuthority(Protocol):
    base_channels: tuple[str, ...]

    def rebuild(
        self,
        base_sequence: np.ndarray,
        lengths: np.ndarray,
        mask: np.ndarray,
        baseline_aggregate: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Rebuild canonical 47-channel sequence and 165 aggregate inputs."""


class CanonicalOULADFeatureAuthority:
    """Thin adapter over the canonical feature functions already used in training."""

    def __init__(self) -> None:
        from src.pipelines import oulad

        if tuple(oulad.BASE_CHANNELS) != BASE_CHANNELS:
            raise ContractValidationError(
                "canonical OULAD base-channel authority changed"
            )
        self._oulad = oulad
        self.base_channels = BASE_CHANNELS

    def rebuild(
        self,
        base_sequence: np.ndarray,
        lengths: np.ndarray,
        mask: np.ndarray,
        baseline_aggregate: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        dynamic = self._oulad._dynamic(base_sequence, mask.astype(bool))
        aggregate_base = self._oulad._aggregate(base_sequence, lengths)
        context = baseline_aggregate[:, 161:]
        aggregate = np.column_stack([aggregate_base, context]).astype(
            np.float32
        )
        if dynamic.shape[2] != 47 or aggregate.shape[1] != 165:
            raise ContractValidationError(
                "canonical OULAD feature dimensions changed"
            )
        return dynamic.astype(np.float32), aggregate


@dataclass(frozen=True)
class OULADTensorSimulation:
    scenario: CounterfactualScenario
    model_inputs: dict[str, torch.Tensor] | None

    def __post_init__(self) -> None:
        simulated = self.scenario.status is SimulationStatus.SIMULATED
        if simulated != (self.model_inputs is not None):
            raise ContractValidationError(
                "tensor simulation inputs and status are inconsistent"
            )


class OULADTensorCounterfactualSimulator:
    def __init__(
        self,
        catalog: OULADTensorEffectCatalog,
        feature_authority: OULADFeatureAuthority | None = None,
    ) -> None:
        self.catalog = catalog
        self.feature_authority = (
            feature_authority or CanonicalOULADFeatureAuthority()
        )
        if tuple(self.feature_authority.base_channels) != BASE_CHANNELS:
            raise ContractValidationError(
                "feature authority base-channel contract mismatch"
            )
        self._channel_index = {
            name: index for index, name in enumerate(BASE_CHANNELS)
        }

    def simulate(
        self,
        action_id: str,
        model_inputs: Mapping[str, torch.Tensor],
        reference_values: Mapping[str, float],
    ) -> OULADTensorSimulation:
        self._validate_inputs(model_inputs)
        spec = self.catalog.by_id(action_id)
        sequence_tensor = model_inputs["sequence"]
        lengths = model_inputs["lengths"].detach().cpu().numpy().astype(int)
        mask = model_inputs["mask"].detach().cpu().numpy().astype(bool)
        aggregate = model_inputs["aggregate"].detach().cpu().numpy()
        observed_weeks = int(lengths[0])
        baseline_base = (
            sequence_tensor.detach()
            .cpu()
            .numpy()[:, :, : len(BASE_CHANNELS)]
            .astype(np.float32, copy=True)
        )
        summary = canonical_state_items(
            {"observed_weeks": observed_weeks}
        )
        if not spec.scorable:
            scenario = CounterfactualScenario(
                action_id=action_id,
                baseline_state=summary,
                simulated_state=summary,
                changes=(),
                status=SimulationStatus.NOT_SCORABLE,
                reason_codes=(str(spec.non_scorable_reason),),
                simulator_version=self.catalog.version,
            )
            return OULADTensorSimulation(scenario=scenario, model_inputs=None)

        modified = baseline_base.copy()
        start = max(0, observed_weeks - spec.lookback_weeks)
        recent = slice(start, observed_weeks)
        for effect in spec.effects:
            channel = self._channel_index[effect.channel_name]
            if effect.operator is TensorEffectOperator.MAX_REFERENCE:
                if effect.reference_key not in reference_values:
                    raise ContractValidationError(
                        f"missing training reference {effect.reference_key}"
                    )
                target = float(reference_values[effect.reference_key])
                modified[0, recent, channel] = np.maximum(
                    modified[0, recent, channel],
                    target,
                )
            elif effect.operator is TensorEffectOperator.ADD:
                assert effect.amount is not None
                modified[0, observed_weeks - 1, channel] += effect.amount
            else:  # pragma: no cover - enum exhaustiveness
                raise ContractValidationError(
                    f"unsupported tensor effect {effect.operator}"
                )

        self._enforce_click_consistency(modified, start, observed_weeks)
        self._recompute_inactivity(modified, start, observed_weeks)
        changes = self._summarize_changes(
            baseline_base,
            modified,
            observed_weeks,
        )
        if not changes:
            scenario = CounterfactualScenario(
                action_id=action_id,
                baseline_state=summary,
                simulated_state=summary,
                changes=(),
                status=SimulationStatus.NO_CHANGE,
                reason_codes=("STATE_ALREADY_MEETS_TENSOR_TARGET",),
                simulator_version=self.catalog.version,
            )
            return OULADTensorSimulation(scenario=scenario, model_inputs=None)

        self._validate_protected_channels(
            baseline_base,
            modified,
            observed_weeks,
        )
        dynamic, rebuilt_aggregate = self.feature_authority.rebuild(
            modified,
            lengths,
            mask,
            aggregate,
        )
        rebuilt = {
            "sequence": torch.as_tensor(
                dynamic,
                dtype=sequence_tensor.dtype,
                device=sequence_tensor.device,
            ),
            "lengths": model_inputs["lengths"].detach().clone(),
            "mask": model_inputs["mask"].detach().clone(),
            "aggregate": torch.as_tensor(
                rebuilt_aggregate,
                dtype=model_inputs["aggregate"].dtype,
                device=model_inputs["aggregate"].device,
            ),
            "static": model_inputs["static"].detach().clone(),
        }
        baseline_summary = {"observed_weeks": observed_weeks}
        simulated_summary = {"observed_weeks": observed_weeks}
        for change in changes:
            baseline_summary[change.feature_name] = change.before
            simulated_summary[change.feature_name] = change.after
        scenario = CounterfactualScenario(
            action_id=action_id,
            baseline_state=canonical_state_items(baseline_summary),
            simulated_state=canonical_state_items(simulated_summary),
            changes=changes,
            status=SimulationStatus.SIMULATED,
            reason_codes=(
                "CANONICAL_OULAD_TENSOR_REBUILT",
                "MODEL_INPUT_COUNTERFACTUAL_ONLY",
            ),
            simulator_version=self.catalog.version,
        )
        return OULADTensorSimulation(
            scenario=scenario,
            model_inputs=rebuilt,
        )

    @staticmethod
    def _validate_inputs(model_inputs: Mapping[str, torch.Tensor]) -> None:
        if set(model_inputs) != MODEL_INPUT_KEYS:
            raise ContractValidationError(
                f"model inputs must be exactly {sorted(MODEL_INPUT_KEYS)}"
            )
        sequence = model_inputs["sequence"]
        lengths = model_inputs["lengths"]
        mask = model_inputs["mask"]
        aggregate = model_inputs["aggregate"]
        static = model_inputs["static"]
        if sequence.ndim != 3 or sequence.shape[0] != 1:
            raise ContractValidationError(
                "counterfactual scoring requires one [1,T,47] sequence"
            )
        if sequence.shape[2] != 47:
            raise ContractValidationError("OULAD sequence must have 47 channels")
        if lengths.shape != (1,) or mask.shape != sequence.shape[:2]:
            raise ContractValidationError("lengths/mask do not align with sequence")
        if aggregate.shape != (1, 165) or static.shape[0] != 1:
            raise ContractValidationError(
                "aggregate/static tensors do not align with one student"
            )
        observed_weeks = int(lengths[0].detach().cpu().item())
        if observed_weeks <= 0 or observed_weeks > sequence.shape[1]:
            raise ContractValidationError("invalid observed sequence length")

    def _enforce_click_consistency(
        self,
        sequence: np.ndarray,
        start: int,
        stop: int,
    ) -> None:
        total = self._channel_index["total_clicks"]
        component_channels = (
            "content_clicks",
            "forum_clicks",
            "quiz_clicks",
            "assessment_related_clicks",
        )
        component_max = np.maximum.reduce(
            [
                sequence[0, start:stop, self._channel_index[name]]
                for name in component_channels
            ]
        )
        sequence[0, start:stop, total] = np.maximum(
            sequence[0, start:stop, total],
            component_max,
        )
        active = self._channel_index["active_days"]
        sequence[0, start:stop, active] = np.clip(
            sequence[0, start:stop, active],
            0.0,
            7.0,
        )

    def _recompute_inactivity(
        self,
        sequence: np.ndarray,
        start: int,
        stop: int,
    ) -> None:
        total = self._channel_index["total_clicks"]
        active = self._channel_index["active_days"]
        days = self._channel_index["days_since_last_vle_activity"]
        weeks = self._channel_index["weeks_without_activity"]
        for week in range(start, stop):
            has_activity = (
                sequence[0, week, total] > 0.0
                or sequence[0, week, active] > 0.0
            )
            if has_activity:
                sequence[0, week, days] = 0.0
                sequence[0, week, weeks] = 0.0
                continue
            previous_days = sequence[0, week - 1, days] if week > 0 else 0.0
            previous_weeks = (
                sequence[0, week - 1, weeks] if week > 0 else 0.0
            )
            sequence[0, week, days] = previous_days + 7.0
            sequence[0, week, weeks] = previous_weeks + 1.0

    def _validate_protected_channels(
        self,
        baseline: np.ndarray,
        simulated: np.ndarray,
        observed_weeks: int,
    ) -> None:
        for name in self.catalog.protected_channels:
            index = self._channel_index[name]
            if not np.array_equal(
                baseline[:, :observed_weeks, index],
                simulated[:, :observed_weeks, index],
            ):
                raise ContractValidationError(
                    f"counterfactual changed protected channel {name}"
                )

    def _summarize_changes(
        self,
        baseline: np.ndarray,
        simulated: np.ndarray,
        observed_weeks: int,
    ) -> tuple[FeatureChange, ...]:
        changes: list[FeatureChange] = []
        for index, name in enumerate(BASE_CHANNELS):
            before_values = baseline[0, :observed_weeks, index]
            after_values = simulated[0, :observed_weeks, index]
            if np.array_equal(before_values, after_values):
                continue
            before = float(before_values.sum())
            after = float(after_values.sum())
            changes.append(
                FeatureChange(
                    feature_name=f"sequence.{name}.observed_sum",
                    before=before,
                    after=after,
                    operation="canonical_tensor_rebuild",
                )
            )
        return tuple(changes)


class HybridPredictionAuthority(Protocol):
    def predict(self, inputs: Mapping[str, torch.Tensor]) -> Any:
        """Return the frozen HybridPredictionOutput contract."""


class FrozenHybridTensorRiskPredictor:
    def __init__(self, authority: HybridPredictionAuthority) -> None:
        self.authority = authority

    def predict_inputs(
        self,
        model_inputs: Mapping[str, torch.Tensor],
    ) -> RiskEstimate:
        output = self.authority.predict(model_inputs)
        risk = float(output.probabilities[0, 1].detach().cpu().item())
        raw_uncertainty = float(output.uncertainty[0].detach().cpu().item())
        normalized_uncertainty = min(
            1.0,
            max(0.0, raw_uncertainty / math.log(2.0)),
        )
        architecture = str(getattr(output, "architecture_hash", "UNKNOWN"))
        return RiskEstimate(
            risk_probability=risk,
            uncertainty=normalized_uncertainty,
            source=f"FROZEN_HYBRID_CNN_BILSTM:{architecture}",
        )


class OULADCounterfactualScorer:
    def __init__(
        self,
        simulator: OULADTensorCounterfactualSimulator,
        predictor: FrozenHybridTensorRiskPredictor,
        ranker: CounterfactualUtilityRanker | None = None,
    ) -> None:
        self.simulator = simulator
        self.predictor = predictor
        self.ranker = ranker or CounterfactualUtilityRanker(
            config=simulator.catalog.utility_config
        )

    def score(
        self,
        *,
        candidate_action_ids: Sequence[str],
        model_inputs: Mapping[str, torch.Tensor],
        reference_values: Mapping[str, float],
        workload_minutes: Mapping[str, int],
        evidence_strength: Mapping[str, float] | None = None,
    ) -> CounterfactualRankingResult:
        unique_ids = tuple(dict.fromkeys(candidate_action_ids))
        baseline = self.predictor.predict_inputs(model_inputs)
        scenarios: dict[str, CounterfactualScenario] = {}
        estimates: dict[str, RiskEstimate] = {}
        for action_id in unique_ids:
            simulation = self.simulator.simulate(
                action_id,
                model_inputs,
                reference_values,
            )
            scenarios[action_id] = simulation.scenario
            if simulation.model_inputs is not None:
                estimates[action_id] = self.predictor.predict_inputs(
                    simulation.model_inputs
                )
        return self.ranker.rank_precomputed(
            candidate_action_ids=unique_ids,
            baseline_estimate=baseline,
            scenarios=scenarios,
            counterfactual_estimates=estimates,
            workload_minutes=workload_minutes,
            evidence_strength=evidence_strength,
        )


__all__ = [
    "BASE_CHANNELS",
    "CanonicalOULADFeatureAuthority",
    "FrozenHybridTensorRiskPredictor",
    "OULADCounterfactualScorer",
    "OULADFeatureAuthority",
    "OULADTensorCounterfactualSimulator",
    "OULADTensorEffectCatalog",
    "OULADTensorSimulation",
    "TensorActionEffectSpec",
    "TensorChannelEffect",
    "TensorEffectOperator",
]
