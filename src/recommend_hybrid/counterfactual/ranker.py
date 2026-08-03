"""Model-estimated risk-reduction ranking for eligible learning actions."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Protocol

from src.recommend_hybrid.exceptions import ContractValidationError

from .contracts import (
    ActionUtility,
    CounterfactualRankingResult,
    RiskEstimate,
    SimulationStatus,
    StateValue,
    UtilityStatus,
)
from .simulator import CounterfactualStateSimulator


class RiskPredictor(Protocol):
    def predict_risk(self, state: Mapping[str, StateValue]) -> RiskEstimate:
        """Return model-estimated risk for one versioned student state."""


@dataclass(frozen=True)
class CounterfactualUtilityConfig:
    version: str = "counterfactual_utility_v1"
    minimum_risk_reduction: float = 0.01
    workload_scale_minutes: float = 60.0

    def __post_init__(self) -> None:
        if not self.version:
            raise ContractValidationError("utility config version is required")
        if not 0.0 <= self.minimum_risk_reduction <= 1.0:
            raise ContractValidationError(
                "minimum_risk_reduction must be in [0, 1]"
            )
        if self.workload_scale_minutes <= 0.0:
            raise ContractValidationError(
                "workload_scale_minutes must be positive"
            )


class CounterfactualUtilityRanker:
    def __init__(
        self,
        simulator: CounterfactualStateSimulator,
        config: CounterfactualUtilityConfig | None = None,
    ) -> None:
        self.simulator = simulator
        self.config = config or CounterfactualUtilityConfig()

    def rank(
        self,
        *,
        candidate_action_ids: Sequence[str],
        baseline_state: Mapping[str, StateValue],
        reference_values: Mapping[str, float],
        predictor: RiskPredictor,
        workload_minutes: Mapping[str, int],
        evidence_strength: Mapping[str, float] | None = None,
    ) -> CounterfactualRankingResult:
        unique_ids = tuple(dict.fromkeys(candidate_action_ids))
        baseline = predictor.predict_risk(baseline_state)
        strengths = evidence_strength or {}
        ranked: list[ActionUtility] = []
        rejected: list[ActionUtility] = []

        for action_id in unique_ids:
            if action_id not in workload_minutes:
                raise ContractValidationError(f"missing workload for {action_id}")
            evidence = float(strengths.get(action_id, 1.0))
            if not 0.0 <= evidence <= 1.0:
                raise ContractValidationError(
                    f"invalid evidence strength for {action_id}"
                )
            scenario = self.simulator.simulate(
                action_id,
                baseline_state,
                reference_values,
            )
            if scenario.status is not SimulationStatus.SIMULATED:
                rejected.append(
                    self._rejected_without_prediction(
                        action_id=action_id,
                        baseline=baseline,
                        workload=int(workload_minutes[action_id]),
                        evidence=evidence,
                        reasons=scenario.reason_codes,
                    )
                )
                continue

            counterfactual = predictor.predict_risk(
                scenario.simulated_mapping()
            )
            reduction = (
                baseline.risk_probability - counterfactual.risk_probability
            )
            uncertainty_penalty = 1.0 - max(
                baseline.uncertainty,
                counterfactual.uncertainty,
            )
            positive_reduction = max(0.0, reduction)
            workload_factor = 1.0 + (
                int(workload_minutes[action_id])
                / self.config.workload_scale_minutes
            )
            utility = (
                positive_reduction
                * evidence
                * uncertainty_penalty
                / workload_factor
            )
            reasons = ["MODEL_ESTIMATED_RISK_REDUCTION"]
            status = UtilityStatus.RANKED
            if reduction < self.config.minimum_risk_reduction:
                status = UtilityStatus.REJECTED
                if reduction < 0.0:
                    reasons = ["MODEL_ESTIMATED_RISK_INCREASE"]
                else:
                    reasons = [
                        "INSUFFICIENT_MODEL_ESTIMATED_RISK_REDUCTION"
                    ]
                utility = 0.0

            item = ActionUtility(
                action_id=action_id,
                status=status,
                baseline_risk=baseline.risk_probability,
                counterfactual_risk=counterfactual.risk_probability,
                risk_reduction=reduction,
                evidence_strength=evidence,
                uncertainty_penalty=uncertainty_penalty,
                workload_minutes=int(workload_minutes[action_id]),
                utility_score=utility,
                changes=scenario.changes,
                reason_codes=tuple(reasons),
            )
            (ranked if status is UtilityStatus.RANKED else rejected).append(
                item
            )

        ranked.sort(
            key=lambda item: (
                -item.utility_score,
                -item.risk_reduction,
                -item.evidence_strength,
                item.workload_minutes,
                item.action_id,
            )
        )
        rejected.sort(key=lambda item: item.action_id)
        return CounterfactualRankingResult(
            baseline_estimate=baseline,
            ranked_actions=tuple(ranked),
            rejected_actions=tuple(rejected),
            ranker_version=self.config.version,
        )

    @staticmethod
    def _rejected_without_prediction(
        *,
        action_id: str,
        baseline: RiskEstimate,
        workload: int,
        evidence: float,
        reasons: tuple[str, ...],
    ) -> ActionUtility:
        return ActionUtility(
            action_id=action_id,
            status=UtilityStatus.REJECTED,
            baseline_risk=baseline.risk_probability,
            counterfactual_risk=baseline.risk_probability,
            risk_reduction=0.0,
            evidence_strength=evidence,
            uncertainty_penalty=1.0 - baseline.uncertainty,
            workload_minutes=workload,
            utility_score=0.0,
            changes=(),
            reason_codes=reasons,
        )


__all__ = [
    "CounterfactualUtilityConfig",
    "CounterfactualUtilityRanker",
    "RiskPredictor",
]
