"""End-to-end OULAD plan construction with counterfactual ranking fallback."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any

import torch
import yaml

from src.recommend_hybrid.common.plan_contracts import LearningPlan
from src.recommend_hybrid.common.policy_contracts import (
    AutomationStatus,
    DatasetId,
    PolicyRecommendationResult,
    Priority,
)
from src.recommend_hybrid.exceptions import ContractValidationError
from src.recommend_hybrid.oulad.plan_builder import OULADLearningPlanBuilder

from .contracts import CounterfactualRankingResult
from .feature_authority import PreprocessedOULADFeatureAuthority
from .oulad_tensor import (
    FrozenHybridTensorRiskPredictor,
    OULADCounterfactualScorer,
    OULADTensorCounterfactualSimulator,
    OULADTensorEffectCatalog,
)
from .reference_profile import OULADReferenceProfile
from .selector import CounterfactualActionSelector

CLAIM_BOUNDARY = "MODEL_ESTIMATED_RISK_REDUCTION_NOT_CAUSAL_EFFECT"


class CounterfactualPlanStatus(str, Enum):
    COUNTERFACTUAL_SCORED = "COUNTERFACTUAL_SCORED"
    POLICY_FALLBACK = "POLICY_FALLBACK"
    EVALUATION_ONLY = "EVALUATION_ONLY"


@dataclass(frozen=True)
class CounterfactualPlanResult:
    plan: LearningPlan
    status: CounterfactualPlanStatus
    ranking: CounterfactualRankingResult | None
    reference_profile_id: str | None
    fallback_reasons: tuple[str, ...]
    claim_boundary: str = CLAIM_BOUNDARY

    def __post_init__(self) -> None:
        if self.claim_boundary != CLAIM_BOUNDARY:
            raise ContractValidationError("invalid counterfactual claim boundary")
        if self.status is CounterfactualPlanStatus.COUNTERFACTUAL_SCORED:
            if self.ranking is None or self.reference_profile_id is None:
                raise ContractValidationError(
                    "scored plan requires ranking and reference profile"
                )
            if self.fallback_reasons:
                raise ContractValidationError(
                    "scored plan cannot contain fallback reasons"
                )
        if self.status is CounterfactualPlanStatus.POLICY_FALLBACK:
            if not self.fallback_reasons:
                raise ContractValidationError(
                    "policy fallback requires at least one reason"
                )

    def to_dict(self) -> dict[str, Any]:
        return {
            "plan": self.plan.to_dict(),
            "status": self.status.value,
            "ranking": self.ranking.to_dict() if self.ranking else None,
            "reference_profile_id": self.reference_profile_id,
            "fallback_reasons": list(self.fallback_reasons),
            "claim_boundary": self.claim_boundary,
        }


class OULADCounterfactualPlanBuilder:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.planning = yaml.safe_load(
            (root / "configs/recommend_hybrid/planning.yaml").read_text(
                encoding="utf-8"
            )
        )
        self.tensor_catalog_path = (
            root
            / "configs/recommend_hybrid/counterfactual_oulad_tensor.yaml"
        )

    def build(
        self,
        policy_result: PolicyRecommendationResult,
        *,
        course_key: str,
        created_at: str,
        model_inputs: Mapping[str, torch.Tensor] | None = None,
        reference_profile: OULADReferenceProfile | None = None,
        prediction_authority: Any | None = None,
        active_contraindications: tuple[str, ...] = (),
    ) -> CounterfactualPlanResult:
        if policy_result.dataset_id is not DatasetId.OULAD:
            raise ContractValidationError(
                "counterfactual plan builder only supports OULAD"
            )
        if policy_result.automation_status is AutomationStatus.EVALUATION_ONLY:
            plan = self._policy_plan(
                policy_result,
                course_key=course_key,
                created_at=created_at,
                active_contraindications=active_contraindications,
            )
            return CounterfactualPlanResult(
                plan=plan,
                status=CounterfactualPlanStatus.EVALUATION_ONLY,
                ranking=None,
                reference_profile_id=None,
                fallback_reasons=("FINAL_STAGE_HAS_NO_INTERVENTION",),
            )

        missing = []
        if model_inputs is None:
            missing.append("MISSING_MODEL_INPUTS")
        if reference_profile is None:
            missing.append("MISSING_TRAINING_REFERENCE_PROFILE")
        if prediction_authority is None:
            missing.append("MISSING_FROZEN_PREDICTION_AUTHORITY")
        if policy_result.automation_status is AutomationStatus.ABSTAIN:
            missing.append("POLICY_ABSTAINED")
        if missing:
            return self._fallback(
                policy_result,
                course_key=course_key,
                created_at=created_at,
                active_contraindications=active_contraindications,
                reasons=tuple(missing),
            )

        assert model_inputs is not None
        assert reference_profile is not None
        self._validate_reference(
            policy_result,
            course_key=course_key,
            profile=reference_profile,
            prediction_authority=prediction_authority,
        )
        decisions = tuple(
            item
            for item in policy_result.action_decisions
            if item.priority is not Priority.NOT_APPLICABLE
            and item.supporting_evidence
        )
        if not decisions:
            return self._fallback(
                policy_result,
                course_key=course_key,
                created_at=created_at,
                active_contraindications=active_contraindications,
                reasons=("NO_POLICY_ELIGIBLE_ACTION_WITH_EVIDENCE",),
                reference_profile_id=reference_profile.profile_id,
            )

        action_ids = tuple(item.action_id for item in decisions)
        metadata = self.planning["action_metadata"]
        workload = {
            action_id: int(metadata[action_id]["weekly_minutes"])
            for action_id in action_ids
        }
        evidence_strength = {
            item.action_id: self._evidence_strength(item)
            for item in decisions
        }
        tensor_catalog = OULADTensorEffectCatalog.load(
            self.tensor_catalog_path
        )
        feature_authority = PreprocessedOULADFeatureAuthority(
            prediction_authority
        )
        scorer = OULADCounterfactualScorer(
            OULADTensorCounterfactualSimulator(
                tensor_catalog,
                feature_authority,
            ),
            FrozenHybridTensorRiskPredictor(prediction_authority),
        )
        ranking = scorer.score(
            candidate_action_ids=action_ids,
            model_inputs=model_inputs,
            reference_values=reference_profile.values(),
            workload_minutes=workload,
            evidence_strength=evidence_strength,
        )
        ranked_ids = tuple(
            item.action_id for item in ranking.ranked_actions
        )
        if not ranked_ids:
            return self._fallback(
                policy_result,
                course_key=course_key,
                created_at=created_at,
                active_contraindications=active_contraindications,
                reasons=("NO_ACTION_MET_MINIMUM_RISK_REDUCTION",),
                ranking=ranking,
                reference_profile_id=reference_profile.profile_id,
            )

        preferred = self._expand_prerequisites(
            ranked_ids,
            eligible_action_ids=set(action_ids),
        )
        builder = OULADLearningPlanBuilder(self.planning)
        builder.solver.selector = CounterfactualActionSelector(
            builder.solver.selector,
            preferred,
        )
        plan = builder.build(
            policy_result,
            course_key=course_key,
            created_at=created_at,
            active_contraindications=active_contraindications,
        )
        return CounterfactualPlanResult(
            plan=plan,
            status=CounterfactualPlanStatus.COUNTERFACTUAL_SCORED,
            ranking=ranking,
            reference_profile_id=reference_profile.profile_id,
            fallback_reasons=(),
        )

    def _policy_plan(
        self,
        policy_result: PolicyRecommendationResult,
        *,
        course_key: str,
        created_at: str,
        active_contraindications: tuple[str, ...],
    ) -> LearningPlan:
        return OULADLearningPlanBuilder(self.planning).build(
            policy_result,
            course_key=course_key,
            created_at=created_at,
            active_contraindications=active_contraindications,
        )

    def _fallback(
        self,
        policy_result: PolicyRecommendationResult,
        *,
        course_key: str,
        created_at: str,
        active_contraindications: tuple[str, ...],
        reasons: tuple[str, ...],
        ranking: CounterfactualRankingResult | None = None,
        reference_profile_id: str | None = None,
    ) -> CounterfactualPlanResult:
        return CounterfactualPlanResult(
            plan=self._policy_plan(
                policy_result,
                course_key=course_key,
                created_at=created_at,
                active_contraindications=active_contraindications,
            ),
            status=CounterfactualPlanStatus.POLICY_FALLBACK,
            ranking=ranking,
            reference_profile_id=reference_profile_id,
            fallback_reasons=reasons,
        )

    def _expand_prerequisites(
        self,
        ranked_action_ids: tuple[str, ...],
        *,
        eligible_action_ids: set[str],
    ) -> tuple[str, ...]:
        metadata = self.planning["action_metadata"]
        ordered: list[str] = []
        visiting: set[str] = set()

        def visit(action_id: str) -> None:
            if action_id in ordered or action_id not in eligible_action_ids:
                return
            if action_id in visiting:
                raise ContractValidationError(
                    "counterfactual action prerequisites contain a cycle"
                )
            visiting.add(action_id)
            for dependency in metadata[action_id]["prerequisites"]:
                visit(str(dependency))
            visiting.remove(action_id)
            ordered.append(action_id)

        for action_id in ranked_action_ids:
            visit(action_id)
        return tuple(ordered)

    @staticmethod
    def _evidence_strength(decision: Any) -> float:
        supporting = len(decision.supporting_evidence)
        missing = len(decision.missing_evidence)
        denominator = supporting + missing
        return supporting / denominator if denominator else 0.0

    @staticmethod
    def _validate_reference(
        policy_result: PolicyRecommendationResult,
        *,
        course_key: str,
        profile: OULADReferenceProfile,
        prediction_authority: Any,
    ) -> None:
        if profile.course_key != course_key:
            raise ContractValidationError(
                "reference profile course does not match request"
            )
        anchor_stage = policy_result.prediction_anchor.anchor_stage
        if anchor_stage is None:
            raise ContractValidationError(
                "counterfactual scoring requires a prediction anchor"
            )
        if _stage_family(profile.stage) != _stage_family(anchor_stage):
            raise ContractValidationError(
                "reference profile stage does not match prediction anchor"
            )
        authority_fold = getattr(prediction_authority, "fold", None)
        if authority_fold is not None and int(authority_fold) != profile.fold:
            raise ContractValidationError(
                "reference profile fold does not match prediction authority"
            )


def _stage_family(stage: str) -> str:
    value = str(stage).upper()
    for family in ("EARLY_20", "EARLY_35", "MIDDLE_50", "LATE_75"):
        if family in value:
            return family
    return value


__all__ = [
    "CLAIM_BOUNDARY",
    "CounterfactualPlanResult",
    "CounterfactualPlanStatus",
    "OULADCounterfactualPlanBuilder",
]
