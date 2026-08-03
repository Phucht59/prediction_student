from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import torch

from src.recommend_hybrid.common.policy_contracts import (
    DatasetId,
    PolicyPredictionContext,
)
from src.recommend_hybrid.counterfactual import (
    CounterfactualPlanStatus,
    OULADCounterfactualPlanBuilder,
    OULADReferenceProfileBuilder,
)
from src.recommend_hybrid.counterfactual.oulad_tensor import BASE_CHANNELS
from src.recommend_hybrid.oulad.policy import RecommendHybridOULAD


class ClickRiskAuthority:
    fold = 0

    def predict(self, inputs):
        total_index = BASE_CHANNELS.index("total_clicks")
        clicks = float(inputs["sequence"][0, :, total_index].sum().item())
        risk = min(0.95, max(0.05, 0.90 - clicks / 500.0))
        return SimpleNamespace(
            probabilities=torch.tensor(
                [[1.0 - risk, risk]],
                dtype=torch.float32,
            ),
            uncertainty=torch.tensor([0.10], dtype=torch.float32),
            architecture_hash="a" * 64,
        )


def _prediction():
    return PolicyPredictionContext(
        dataset_id=DatasetId.OULAD,
        predicted_class=1,
        class_probabilities=(0.30, 0.70),
        confidence=0.70,
        uncertainty=0.45,
        seed_disagreement=0.03,
        checkpoint_lineage=("frozen_cnn_bilstm_seed_ensemble",),
        architecture_authority="RECOMMEND_HYBRID_MODEL_AUTHORITY",
        representation_lineage=(
            "student_state_embedding:64",
            "tabular_expert_embedding:32",
        ),
    )


def _policy_result(root):
    return RecommendHybridOULAD(root).recommend(
        student_key="counterfactual-test",
        course_key="AAA-2014J",
        requested_cutoff=63,
        prediction=_prediction(),
        max_observation_cutoff=62,
        activity_level=4,
        recent_activity_trend=-5,
        inactivity_streak=14,
        assessment_progress=0.40,
        assessments_due=2,
    )


def _model_inputs():
    sequence = torch.zeros((1, 4, 47), dtype=torch.float32)
    index = {name: i for i, name in enumerate(BASE_CHANNELS)}
    sequence[0, :, index["total_clicks"]] = torch.tensor(
        [12.0, 8.0, 0.0, 0.0]
    )
    sequence[0, :, index["active_days"]] = torch.tensor(
        [2.0, 1.0, 0.0, 0.0]
    )
    sequence[0, :, index["days_since_last_vle_activity"]] = torch.tensor(
        [0.0, 0.0, 7.0, 14.0]
    )
    sequence[0, :, index["weeks_without_activity"]] = torch.tensor(
        [0.0, 0.0, 1.0, 2.0]
    )
    sequence[0, :, index["score_missing_mask"]] = 1.0
    aggregate = torch.zeros((1, 165), dtype=torch.float32)
    aggregate[0, 161:] = torch.tensor([0.50, 4.0, 8.0, 0.25])
    return {
        "sequence": sequence,
        "lengths": torch.tensor([4], dtype=torch.int64),
        "mask": torch.ones((1, 4), dtype=torch.float32),
        "aggregate": aggregate,
        "static": torch.zeros((1, 13), dtype=torch.float32),
    }


def _reference_profile():
    sequence = np.zeros((3, 4, 47), dtype=np.float32)
    index = {name: i for i, name in enumerate(BASE_CHANNELS)}
    for row, multiplier in enumerate((1.0, 1.2, 1.4)):
        sequence[row, :, index["total_clicks"]] = (
            np.array([35, 45, 55, 65]) * multiplier
        )
        sequence[row, :, index["active_days"]] = [2, 3, 4, 5]
        sequence[row, :, index["content_clicks"]] = (
            np.array([15, 20, 25, 30]) * multiplier
        )
        sequence[row, :, index["unique_sites"]] = [2, 3, 4, 5]
        sequence[row, :, index["quiz_clicks"]] = [4, 6, 8, 10]
        sequence[row, :, index["assessment_related_clicks"]] = [
            5,
            7,
            9,
            11,
        ]
    return OULADReferenceProfileBuilder(
        minimum_positive_observations=1
    ).build(
        sequence=sequence,
        lengths=np.array([4, 4, 4]),
        fold=0,
        stage="M1_MIDDLE_50PCT",
        course_key="AAA-2014J",
    )


def test_counterfactual_builder_scores_and_constrains_plan(root):
    result = OULADCounterfactualPlanBuilder(root).build(
        _policy_result(root),
        course_key="AAA-2014J",
        created_at="2026-08-03T20:30:00Z",
        model_inputs=_model_inputs(),
        reference_profile=_reference_profile(),
        prediction_authority=ClickRiskAuthority(),
    )
    assert result.status is CounterfactualPlanStatus.COUNTERFACTUAL_SCORED
    assert result.ranking is not None
    assert result.ranking.ranked_actions
    assert result.plan.selected_actions
    ranked = {item.action_id for item in result.ranking.ranked_actions}
    selected = {item.action_id for item in result.plan.selected_actions}
    assert ranked & selected
    assert result.claim_boundary.endswith("NOT_CAUSAL_EFFECT")


def test_counterfactual_builder_falls_back_when_inputs_are_missing(root):
    result = OULADCounterfactualPlanBuilder(root).build(
        _policy_result(root),
        course_key="AAA-2014J",
        created_at="2026-08-03T20:30:00Z",
    )
    assert result.status is CounterfactualPlanStatus.POLICY_FALLBACK
    assert "MISSING_MODEL_INPUTS" in result.fallback_reasons
    assert "MISSING_TRAINING_REFERENCE_PROFILE" in result.fallback_reasons
    assert "MISSING_FROZEN_PREDICTION_AUTHORITY" in result.fallback_reasons
