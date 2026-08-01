"""Dataset-routed Phase 3 policy to Phase 4 learning-plan pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import yaml

from .common.plan_contracts import LearningPlan
from .common.policy_contracts import DatasetId, PolicyPredictionContext
from .oulad.plan_builder import OULADLearningPlanBuilder
from .oulad.policy import RecommendHybridOULAD
from .uci.plan_builder import UCILearningPlanBuilder
from .uci.policy import RecommendHybridUCI


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


@dataclass(frozen=True)
class UCIPlanRequest:
    dataset_id: DatasetId
    student_key: str
    course_key: str
    prediction: PolicyPredictionContext
    g1: float | None
    g2: float | None
    absences: int | None
    study_time: int | None
    previous_failures: int | None
    next_assessment_available: bool | None
    requested_cutoff: float | None = None
    stage_evidence_known: bool = True
    extra_features: dict | None = None
    active_contraindications: tuple[str, ...] = ()
    created_at: str = field(default_factory=_utc_now)

    def __post_init__(self) -> None:
        if self.dataset_id not in {DatasetId.STUDENT_MAT, DatasetId.STUDENT_POR}:
            raise ValueError("UCI plan request requires student_mat or student_por")


@dataclass(frozen=True)
class OULADPlanRequest:
    student_key: str
    course_key: str
    requested_cutoff: float
    prediction: PolicyPredictionContext | None
    max_observation_cutoff: float | None = None
    activity_level: float | None = None
    recent_activity_trend: float | None = None
    inactivity_streak: int | None = None
    assessment_progress: float | None = None
    assessments_due: int | None = None
    grade_trend: float | None = None
    grade_release_verified: bool = False
    knowledge_gap: str | None = None
    active_contraindications: tuple[str, ...] = ()
    created_at: str = field(default_factory=_utc_now)


PlanRequest = UCIPlanRequest | OULADPlanRequest


class RecommendHybridPipeline:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.planning = yaml.safe_load(
            (root / "configs/recommend_hybrid/planning.yaml").read_text(encoding="utf-8")
        )
        self.uci_builders = {
            DatasetId.STUDENT_MAT: (
                RecommendHybridUCI(root, DatasetId.STUDENT_MAT),
                UCILearningPlanBuilder(self.planning),
            ),
            DatasetId.STUDENT_POR: (
                RecommendHybridUCI(root, DatasetId.STUDENT_POR),
                UCILearningPlanBuilder(self.planning),
            ),
        }
        self.oulad_policy = RecommendHybridOULAD(root)
        self.oulad_builder = OULADLearningPlanBuilder(self.planning)

    def generate(self, request: PlanRequest) -> LearningPlan:
        if isinstance(request, UCIPlanRequest):
            return self._generate_uci(request)
        if isinstance(request, OULADPlanRequest):
            return self._generate_oulad(request)
        raise TypeError("unsupported recommend_hybrid plan request")

    def _generate_uci(self, request: UCIPlanRequest) -> LearningPlan:
        policy, builder = self.uci_builders[request.dataset_id]
        result = policy.recommend(
            student_key=request.student_key,
            course_key=request.course_key,
            prediction=request.prediction,
            g1=request.g1,
            g2=request.g2,
            absences=request.absences,
            study_time=request.study_time,
            previous_failures=request.previous_failures,
            next_assessment_available=request.next_assessment_available,
            requested_cutoff=request.requested_cutoff,
            stage_evidence_known=request.stage_evidence_known,
            extra_features=request.extra_features,
        )
        return builder.build(
            result,
            course_key=request.course_key,
            created_at=request.created_at,
            active_contraindications=request.active_contraindications,
        )

    def _generate_oulad(self, request: OULADPlanRequest) -> LearningPlan:
        result = self.oulad_policy.recommend(
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
        return self.oulad_builder.build(
            result,
            course_key=request.course_key,
            created_at=request.created_at,
            active_contraindications=request.active_contraindications,
        )


__all__ = ["OULADPlanRequest", "PlanRequest", "RecommendHybridPipeline", "UCIPlanRequest"]
