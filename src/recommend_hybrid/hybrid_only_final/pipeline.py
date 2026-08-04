"""Public final hybrid-only OULAD recommendation pipeline."""
from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch

from src.recommend_hybrid.counterfactual.reference_profile import OULADReferenceProfile
from src.recommend_hybrid.oulad.policy import RecommendHybridOULAD
from src.recommend_hybrid.pipeline import OULADPlanRequest

from .plan_builder import HybridOnlyFinalPlanBuilder, HybridOnlyFinalPlanResult


class RecommendHybridOnlyFinalPipeline:
    """Generate a released hybrid-only plan with no learned action ranker."""

    def __init__(self, root: Path) -> None:
        self.root = root
        self.policy = RecommendHybridOULAD(root)
        # Construction fails closed until HYBRID_ONLY_OFFLINE_SILVER_VALIDATED.
        self.plan_builder = HybridOnlyFinalPlanBuilder(root)

    def generate(
        self,
        request: OULADPlanRequest,
        *,
        model_inputs: Mapping[str, torch.Tensor] | None = None,
        reference_profile: OULADReferenceProfile | None = None,
        prediction_authority: Any | None = None,
    ) -> HybridOnlyFinalPlanResult:
        policy_result = self.policy.recommend(
            student_key=request.student_key,
            course_key=request.course_key,
            requested_cutoff=request.requested_cutoff,
            prediction=request.prediction,
            max_observation_cutoff=request.max_observation_cutoff,
            activity_level=request.activity_level,
            recent_activity_trend=request.recent_activity_trend,
            inactivity_streak=request.inactivity_streak,
            assessment_progress=request.assessment_progress,
            assessments_due=request.assessments_due,
            grade_trend=request.grade_trend,
            grade_release_verified=request.grade_release_verified,
            knowledge_gap=request.knowledge_gap,
        )
        return self.plan_builder.build(
            policy_result,
            course_key=request.course_key,
            created_at=request.created_at,
            model_inputs=model_inputs,
            reference_profile=reference_profile,
            prediction_authority=prediction_authority,
            active_contraindications=request.active_contraindications,
        )


__all__ = ["RecommendHybridOnlyFinalPipeline"]
