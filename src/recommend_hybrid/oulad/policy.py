"""Config-driven OULAD evidence-action policy with arbitrary-cutoff routing."""

from __future__ import annotations

from pathlib import Path

from src.recommend_hybrid.common.policy_contracts import (
    DatasetId,
    PolicyPredictionContext,
    RecommendationRequest,
    RoutingStatus,
)
from src.recommend_hybrid.common.policy_engine import (
    abstained_result,
    evaluation_only_result,
    load_policy,
    run_declared_policy,
)

from .action_catalog import OULAD_ACTIONS
from .cutoff_router import route_oulad_cutoff
from .evidence_severity import evaluate_oulad_severity
from .observed_state import build_oulad_observed_state


class RecommendHybridOULAD:
    def __init__(self, root: Path) -> None:
        self.common = load_policy(root / "configs/recommend_hybrid/policy_common.yaml")
        self.config = load_policy(root / "configs/recommend_hybrid/policy_oulad.yaml")
        if tuple(self.config["allowed_actions"]) != OULAD_ACTIONS:
            raise ValueError("OULAD action catalog/config mismatch")

    def recommend(
        self,
        *,
        student_key: str,
        course_key: str,
        requested_cutoff: float,
        prediction: PolicyPredictionContext | None,
        max_observation_cutoff: float | None = None,
        activity_level: float | None = None,
        recent_activity_trend: float | None = None,
        inactivity_streak: int | None = None,
        assessment_progress: float | None = None,
        assessments_due: int | None = None,
        grade_trend: float | None = None,
        grade_release_verified: bool = False,
        knowledge_gap: str | None = None,
    ):
        lineage = prediction.checkpoint_lineage if prediction is not None else ()
        anchor = route_oulad_cutoff(
            requested_cutoff,
            checkpoint_lineage=lineage,
            config=self.config,
        )
        if anchor.routing_status is RoutingStatus.NO_VALIDATED_PREDICTION_ANCHOR:
            return abstained_result(
                dataset_id=DatasetId.OULAD,
                student_key=student_key,
                requested_cutoff=requested_cutoff,
                anchor=anchor,
                reasons=("NO_VALIDATED_PREDICTION_ANCHOR",),
                policy_version=self.config["policy_version"],
            )
        if anchor.routing_status is RoutingStatus.EVALUATION_ONLY:
            return evaluation_only_result(
                dataset_id=DatasetId.OULAD,
                student_key=student_key,
                requested_cutoff=requested_cutoff,
                anchor=anchor,
                policy_version=self.config["policy_version"],
            )
        if prediction is None:
            raise ValueError("validated OULAD anchor requires frozen prediction context")
        if prediction.dataset_id is not DatasetId.OULAD:
            raise ValueError("OULAD policy requires OULAD prediction context")
        raw = build_oulad_observed_state(
            requested_cutoff=requested_cutoff,
            max_observation_cutoff=max_observation_cutoff,
            activity_level=activity_level,
            recent_activity_trend=recent_activity_trend,
            inactivity_streak=inactivity_streak,
            assessment_progress=assessment_progress,
            assessments_due=assessments_due,
            grade_trend=grade_trend,
            grade_release_verified=grade_release_verified,
            knowledge_gap=knowledge_gap,
        )
        evidence = evaluate_oulad_severity(raw, self.config)
        request = RecommendationRequest(
            dataset_id=DatasetId.OULAD,
            student_key=student_key,
            course_key=course_key,
            requested_cutoff=requested_cutoff,
            available_assessments=("due_assessments",)
            if assessments_due is not None and assessments_due > 0
            else (),
            prediction_context=prediction,
            observed_state=evidence,
        )
        return run_declared_policy(
            request,
            anchor=anchor,
            stage=anchor.anchor_stage,
            dataset_config=self.config,
            common_config=self.common,
        )


__all__ = ["RecommendHybridOULAD"]
