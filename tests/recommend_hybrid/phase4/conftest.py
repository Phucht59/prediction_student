from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from src.recommend_hybrid.common.policy_contracts import DatasetId, PolicyPredictionContext
from src.recommend_hybrid.pipeline import OULADPlanRequest, RecommendHybridPipeline, UCIPlanRequest

CREATED_AT = "2026-08-01T00:00:00Z"


@pytest.fixture(scope="session")
def phase4_root() -> Path:
    return Path(__file__).resolve().parents[3]


@pytest.fixture(scope="session")
def pipeline(phase4_root) -> RecommendHybridPipeline:
    return RecommendHybridPipeline(phase4_root)


@pytest.fixture()
def uci_prediction() -> PolicyPredictionContext:
    return PolicyPredictionContext(
        dataset_id=DatasetId.STUDENT_MAT,
        predicted_class=0,
        class_probabilities=(0.7, 0.2, 0.1),
        confidence=0.7,
        uncertainty=0.45,
        seed_disagreement=0.03,
        checkpoint_lineage=("frozen_uci_cnn_bilstm_seed_ensemble",),
        architecture_authority="RECOMMEND_HYBRID_MODEL_AUTHORITY",
    )


@pytest.fixture()
def oulad_prediction() -> PolicyPredictionContext:
    return PolicyPredictionContext(
        dataset_id=DatasetId.OULAD,
        predicted_class=1,
        class_probabilities=(0.3, 0.7),
        confidence=0.7,
        uncertainty=0.45,
        seed_disagreement=0.03,
        checkpoint_lineage=("recommend_hybrid_intervention_outer_0_seed_ensemble",),
        architecture_authority="RECOMMEND_HYBRID_MODEL_AUTHORITY",
    )


def uci_request(prediction, dataset_id=DatasetId.STUDENT_MAT, **updates):
    if prediction.dataset_id is not dataset_id:
        prediction = replace(prediction, dataset_id=dataset_id)
    values = dict(
        dataset_id=dataset_id,
        student_key="uci-student",
        course_key="uci-course",
        prediction=prediction,
        g1=8,
        g2=None,
        absences=12,
        study_time=1,
        previous_failures=1,
        next_assessment_available=True,
        created_at=CREATED_AT,
    )
    values.update(updates)
    return UCIPlanRequest(**values)


def oulad_request(prediction, cutoff=50, **updates):
    values = dict(
        student_key="oulad-student",
        course_key="oulad-course",
        requested_cutoff=cutoff,
        prediction=prediction,
        max_observation_cutoff=cutoff - 1 if cutoff not in (0, 100) else None,
        activity_level=4.0,
        recent_activity_trend=-6.0,
        inactivity_streak=14,
        assessment_progress=0.2,
        assessments_due=2,
        knowledge_gap="topic-A",
        created_at=CREATED_AT,
    )
    values.update(updates)
    return OULADPlanRequest(**values)
