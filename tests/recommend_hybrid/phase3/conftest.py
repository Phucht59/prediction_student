from __future__ import annotations

from pathlib import Path

import pytest

from src.recommend_hybrid.common.policy_contracts import DatasetId, PolicyPredictionContext
from src.recommend_hybrid.oulad.policy import RecommendHybridOULAD
from src.recommend_hybrid.uci.policy import RecommendHybridUCI


@pytest.fixture(scope="session")
def phase3_root() -> Path:
    return Path(__file__).resolve().parents[3]


@pytest.fixture()
def oulad_prediction() -> PolicyPredictionContext:
    return PolicyPredictionContext(
        dataset_id=DatasetId.OULAD,
        predicted_class=1,
        class_probabilities=(0.30, 0.70),
        confidence=0.70,
        uncertainty=0.45,
        seed_disagreement=0.03,
        checkpoint_lineage=("recommend_hybrid_intervention_outer_0_seed_ensemble",),
        architecture_authority="RECOMMEND_HYBRID_MODEL_AUTHORITY",
        representation_lineage=("student_state_embedding:64", "tabular_expert_embedding:32"),
    )


@pytest.fixture()
def uci_prediction() -> PolicyPredictionContext:
    return PolicyPredictionContext(
        dataset_id=DatasetId.STUDENT_MAT,
        predicted_class=0,
        class_probabilities=(0.70, 0.20, 0.10),
        confidence=0.70,
        uncertainty=0.45,
        seed_disagreement=0.03,
        checkpoint_lineage=("frozen_uci_cnn_bilstm_seed_ensemble",),
        architecture_authority="FINAL_THESIS_MODEL_AUTHORITY",
        representation_lineage=("student_state_embedding:64", "tabular_expert_embedding:32"),
    )


@pytest.fixture()
def uci_mat_policy(phase3_root):
    return RecommendHybridUCI(phase3_root, DatasetId.STUDENT_MAT)


@pytest.fixture()
def uci_por_policy(phase3_root):
    return RecommendHybridUCI(phase3_root, DatasetId.STUDENT_POR)


@pytest.fixture()
def oulad_policy(phase3_root):
    return RecommendHybridOULAD(phase3_root)


def decision(result, action_id):
    return next(item for item in result.action_decisions if item.action_id == action_id)
