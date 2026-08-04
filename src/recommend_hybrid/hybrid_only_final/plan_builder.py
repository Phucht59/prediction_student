"""Final OULAD plan builder using only frozen hybrid counterfactual evidence."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import torch
import yaml

from src.recommend_hybrid.common.plan_contracts import LearningPlan
from src.recommend_hybrid.common.policy_contracts import (
    EvidenceSeverity,
    PolicyActionDecision,
    PolicyRecommendationResult,
    Priority,
)
from src.recommend_hybrid.counterfactual.plan_builder import (
    CounterfactualPlanStatus,
    OULADCounterfactualPlanBuilder,
)
from src.recommend_hybrid.counterfactual.reference_profile import OULADReferenceProfile
from src.recommend_hybrid.counterfactual.selector import CounterfactualActionSelector
from src.recommend_hybrid.oulad.plan_builder import OULADLearningPlanBuilder

from .runtime import load_released_hybrid_only_config
from .scorer import HybridActionEvidence, HybridOnlyDecision, score_hybrid_actions

CLAIM_BOUNDARY = "HYBRID_MODEL_GUIDED_DECISION_SUPPORT_NOT_CAUSAL_EFFECT"
SEVERITY_NEED = {
    EvidenceSeverity.CRITICAL: 1.00,
    EvidenceSeverity.HIGH: 0.75,
    EvidenceSeverity.MEDIUM: 0.50,
    EvidenceSeverity.LOW: 0.25,
    EvidenceSeverity.NONE: 0.00,
    EvidenceSeverity.MISSING: 0.00,
}


def evidence_need_score(decision: PolicyActionDecision) -> float:
    """Use the same ordinal evidence scale used by offline normalization."""

    if not decision.supporting_evidence:
        return 0.0
    return max(SEVERITY_NEED[item.severity] for item in decision.supporting_evidence)


@dataclass(frozen=True)
class HybridOnlyFinalPlanResult:
    plan: LearningPlan
    status: str
    decision: HybridOnlyDecision | None
    fallback_reasons: tuple[str, ...]
    claim_boundary: str = CLAIM_BOUNDARY

    def to_dict(self) -> dict[str, Any]:
        return {
            "plan": self.plan.to_dict(),
            "status": self.status,
            "selected_action": (
                self.decision.selected_action.action_id
                if self.decision and self.decision.selected_action
                else None
            ),
            "ranked_actions": (
                [item.action_id for item in self.decision.ranked_actions]
                if self.decision
                else []
            ),
            "abstention_reason": (
                self.decision.abstention_reason if self.decision else None
            ),
            "fallback_reasons": list(self.fallback_reasons),
            "claim_boundary": self.claim_boundary,
        }


class HybridOnlyFinalPlanBuilder:
    """Build a plan only when the hybrid-only offline release is authorized."""

    def __init__(self, root: Path) -> None:
        self.root = root
        self.config = load_released_hybrid_only_config(root)
        self.counterfactual = OULADCounterfactualPlanBuilder(root)
        self.planning = yaml.safe_load(
            (root / "configs/recommend_hybrid/planning.yaml").read_text(
                encoding="utf-8"
            )
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
    ) -> HybridOnlyFinalPlanResult:
        raw = self.counterfactual.build(
            policy_result,
            course_key=course_key,
            created_at=created_at,
            model_inputs=model_inputs,
            reference_profile=reference_profile,
            prediction_authority=prediction_authority,
            active_contraindications=active_contraindications,
        )
        if raw.status is CounterfactualPlanStatus.EVALUATION_ONLY:
            return HybridOnlyFinalPlanResult(
                plan=raw.plan,
                status="EVALUATION_ONLY",
                decision=None,
                fallback_reasons=raw.fallback_reasons,
            )
        if raw.ranking is None:
            return HybridOnlyFinalPlanResult(
                plan=raw.plan,
                status="POLICY_FALLBACK",
                decision=None,
                fallback_reasons=raw.fallback_reasons
                or ("HYBRID_COUNTERFACTUAL_EVIDENCE_UNAVAILABLE",),
            )

        decisions = {
            item.action_id: item
            for item in policy_result.action_decisions
            if item.priority is not Priority.NOT_APPLICABLE
            and item.supporting_evidence
        }
        utility_rows = {
            item.action_id: item
            for item in (
                *raw.ranking.ranked_actions,
                *raw.ranking.rejected_actions,
            )
        }
        candidates = []
        metadata = self.planning["action_metadata"]
        for action_id, policy_decision in decisions.items():
            utility = utility_rows.get(action_id)
            if utility is None:
                continue
            candidates.append(
                HybridActionEvidence(
                    action_id=action_id,
                    risk_reduction=float(utility.risk_reduction),
                    evidence_strength=self.counterfactual._evidence_strength(
                        policy_decision
                    ),
                    need_score=evidence_need_score(policy_decision),
                    uncertainty=float(1.0 - utility.uncertainty_penalty),
                    workload_minutes=int(metadata[action_id]["weekly_minutes"]),
                    available=True,
                    prerequisite_met=True,
                    contraindicated=bool(
                        set(metadata[action_id]["contraindications"])
                        & set(active_contraindications)
                    ),
                )
            )
        decision = score_hybrid_actions(candidates, self.config)
        if not decision.issued:
            return HybridOnlyFinalPlanResult(
                plan=raw.plan,
                status="POLICY_FALLBACK",
                decision=decision,
                fallback_reasons=(
                    decision.abstention_reason
                    or "HYBRID_ONLY_SELECTIVE_ABSTENTION",
                ),
            )

        ranked_ids = tuple(item.action_id for item in decision.ranked_actions)
        preferred = self.counterfactual._expand_prerequisites(
            ranked_ids,
            eligible_action_ids=set(decisions),
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
        return HybridOnlyFinalPlanResult(
            plan=plan,
            status="HYBRID_ONLY_SCORED",
            decision=decision,
            fallback_reasons=(),
        )


__all__ = [
    "CLAIM_BOUNDARY",
    "HybridOnlyFinalPlanBuilder",
    "HybridOnlyFinalPlanResult",
    "evidence_need_score",
]
