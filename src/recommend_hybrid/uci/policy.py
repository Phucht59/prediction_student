"""Config-driven UCI MAT/POR evidence-action policy."""

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
    load_policy,
    run_declared_policy,
)

from .action_catalog import UCI_ACTIONS
from .evidence_severity import evaluate_uci_severity
from .observed_state import build_uci_observed_state
from .stage_router import route_uci_stage


class RecommendHybridUCI:
    def __init__(self, root: Path, dataset_id: DatasetId) -> None:
        if dataset_id not in {DatasetId.STUDENT_MAT, DatasetId.STUDENT_POR}:
            raise ValueError("UCI policy supports only student_mat and student_por")
        suffix = "mat" if dataset_id is DatasetId.STUDENT_MAT else "por"
        self.common = load_policy(root / "configs/recommend_hybrid/policy_common.yaml")
        self.config = load_policy(root / f"configs/recommend_hybrid/policy_uci_{suffix}.yaml")
        if self.config["dataset_id"] != dataset_id.value:
            raise ValueError("UCI dataset/config mismatch")
        if tuple(self.config["allowed_actions"]) != UCI_ACTIONS:
            raise ValueError("UCI action catalog/config mismatch")
        self.dataset_id = dataset_id

    def recommend(
        self,
        *,
        student_key: str,
        course_key: str,
        prediction: PolicyPredictionContext,
        g1: float | None,
        g2: float | None,
        absences: int | None,
        study_time: int | None,
        previous_failures: int | None,
        next_assessment_available: bool | None,
        requested_cutoff: float | None = None,
        stage_evidence_known: bool = True,
        extra_features: dict | None = None,
    ):
        if prediction.dataset_id is not self.dataset_id:
            raise ValueError("UCI prediction context/config dataset mismatch")
        anchor = route_uci_stage(
            g1=g1,
            g2=g2,
            checkpoint_lineage=prediction.checkpoint_lineage,
            requested_cutoff=requested_cutoff,
            stage_evidence_known=stage_evidence_known,
        )
        if anchor.routing_status is not RoutingStatus.ROUTED:
            return abstained_result(
                dataset_id=self.dataset_id,
                student_key=student_key,
                requested_cutoff=anchor.requested_cutoff,
                anchor=anchor,
                reasons=("INSUFFICIENT_STAGE_EVIDENCE",),
                policy_version=self.config["policy_version"],
            )
        raw = build_uci_observed_state(
            stage=anchor.anchor_stage,
            cutoff=anchor.requested_cutoff,
            g1=g1,
            g2=g2,
            absences=absences,
            study_time=study_time,
            previous_failures=previous_failures,
            next_assessment_available=next_assessment_available,
            extra_features=extra_features,
        )
        evidence = evaluate_uci_severity(raw, self.config)
        available = tuple(
            name for name, value in (("G1", g1), ("G2", g2)) if value is not None
        )
        request = RecommendationRequest(
            dataset_id=self.dataset_id,
            student_key=student_key,
            course_key=course_key,
            requested_cutoff=anchor.requested_cutoff,
            available_assessments=available,
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


__all__ = ["RecommendHybridUCI"]
