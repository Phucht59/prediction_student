from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from src.recommend_hybrid.counterfactual.contracts import SimulationStatus
from src.recommend_hybrid.counterfactual.oulad_tensor import (
    BASE_CHANNELS,
    FrozenHybridTensorRiskPredictor,
    OULADCounterfactualScorer,
    OULADTensorCounterfactualSimulator,
    OULADTensorEffectCatalog,
)

ROOT = Path(__file__).resolve().parents[3]


class FakeFeatureAuthority:
    base_channels = BASE_CHANNELS

    def rebuild(self, base_sequence, lengths, mask, baseline_aggregate):
        dynamic = np.zeros(
            (base_sequence.shape[0], base_sequence.shape[1], 47),
            dtype=np.float32,
        )
        dynamic[:, :, : len(BASE_CHANNELS)] = base_sequence
        aggregate = np.zeros((base_sequence.shape[0], 165), dtype=np.float32)
        aggregate[:, :16] = base_sequence.sum(axis=1)
        aggregate[:, 161:] = baseline_aggregate[:, 161:]
        return dynamic, aggregate


class ClickRiskAuthority:
    def predict(self, inputs):
        total_index = BASE_CHANNELS.index("total_clicks")
        clicks = float(inputs["sequence"][0, :, total_index].sum().item())
        risk = min(0.95, max(0.05, 0.85 - clicks / 500.0))
        probability = torch.tensor(
            [[1.0 - risk, risk]],
            dtype=torch.float32,
        )
        uncertainty = torch.tensor([0.10], dtype=torch.float32)
        return SimpleNamespace(
            probabilities=probability,
            uncertainty=uncertainty,
            architecture_hash="a" * 64,
        )


def _catalog():
    return OULADTensorEffectCatalog.load(
        ROOT / "configs/recommend_hybrid/counterfactual_oulad_tensor.yaml"
    )


def _inputs():
    sequence = torch.zeros((1, 4, 47), dtype=torch.float32)
    index = {name: i for i, name in enumerate(BASE_CHANNELS)}
    sequence[0, :, index["total_clicks"]] = torch.tensor(
        [20.0, 15.0, 0.0, 0.0]
    )
    sequence[0, :, index["active_days"]] = torch.tensor(
        [2.0, 1.0, 0.0, 0.0]
    )
    sequence[0, :, index["available_score_count"]] = 3.0
    sequence[0, :, index["cumulative_mean_score"]] = 70.0
    sequence[0, :, index["cumulative_weighted_score"]] = 28.0
    sequence[0, :, index["score_missing_mask"]] = 1.0
    sequence[0, :, index["days_since_last_vle_activity"]] = torch.tensor(
        [0.0, 0.0, 7.0, 14.0]
    )
    sequence[0, :, index["weeks_without_activity"]] = torch.tensor(
        [0.0, 0.0, 1.0, 2.0]
    )
    aggregate = torch.zeros((1, 165), dtype=torch.float32)
    aggregate[0, 161:] = torch.tensor([0.50, 4.0, 8.0, 0.25])
    return {
        "sequence": sequence,
        "lengths": torch.tensor([4], dtype=torch.int64),
        "mask": torch.ones((1, 4), dtype=torch.float32),
        "aggregate": aggregate,
        "static": torch.zeros((1, 13), dtype=torch.float32),
    }


def _references():
    return {
        "total_clicks_p50": 40.0,
        "total_clicks_p65": 60.0,
        "active_days_p50": 3.0,
        "content_clicks_p50": 25.0,
        "content_clicks_p65": 35.0,
        "unique_sites_p50": 4.0,
        "quiz_clicks_p50": 8.0,
        "quiz_clicks_p65": 12.0,
        "assessment_related_clicks_p50": 10.0,
    }


def test_tensor_simulation_preserves_scores_static_and_context():
    simulator = OULADTensorCounterfactualSimulator(
        _catalog(),
        FakeFeatureAuthority(),
    )
    inputs = _inputs()
    result = simulator.simulate("VLE_ENGAGEMENT", inputs, _references())
    assert result.scenario.status is SimulationStatus.SIMULATED
    assert result.model_inputs is not None
    rebuilt = result.model_inputs
    index = {name: i for i, name in enumerate(BASE_CHANNELS)}
    for channel in (
        "available_score_count",
        "cumulative_mean_score",
        "cumulative_weighted_score",
        "score_missing_mask",
    ):
        i = index[channel]
        assert torch.equal(
            rebuilt["sequence"][0, :, i],
            inputs["sequence"][0, :, i],
        )
    assert torch.equal(rebuilt["static"], inputs["static"])
    assert torch.equal(
        rebuilt["aggregate"][:, 161:],
        inputs["aggregate"][:, 161:],
    )
    assert rebuilt["sequence"][0, 2, index["total_clicks"]] == 60.0
    assert (
        rebuilt["sequence"][0, 3, index["weeks_without_activity"]]
        == 0.0
    )


def test_tensor_non_scorable_action_uses_policy_fallback():
    simulator = OULADTensorCounterfactualSimulator(
        _catalog(),
        FakeFeatureAuthority(),
    )
    result = simulator.simulate(
        "ADVISOR_ESCALATION",
        _inputs(),
        _references(),
    )
    assert result.model_inputs is None
    assert result.scenario.status is SimulationStatus.NOT_SCORABLE


def test_frozen_predictor_normalizes_entropy_uncertainty():
    predictor = FrozenHybridTensorRiskPredictor(ClickRiskAuthority())
    estimate = predictor.predict_inputs(_inputs())
    assert 0.0 <= estimate.risk_probability <= 1.0
    assert 0.0 <= estimate.uncertainty <= 1.0
    assert estimate.source.startswith("FROZEN_HYBRID_CNN_BILSTM")


def test_tensor_scorer_uses_modified_model_inputs_for_ranking():
    simulator = OULADTensorCounterfactualSimulator(
        _catalog(),
        FakeFeatureAuthority(),
    )
    scorer = OULADCounterfactualScorer(
        simulator,
        FrozenHybridTensorRiskPredictor(ClickRiskAuthority()),
    )
    result = scorer.score(
        candidate_action_ids=(
            "STUDY_SCHEDULE",
            "VLE_ENGAGEMENT",
            "ADVISOR_ESCALATION",
        ),
        model_inputs=_inputs(),
        reference_values=_references(),
        workload_minutes={
            "STUDY_SCHEDULE": 30,
            "VLE_ENGAGEMENT": 90,
            "ADVISOR_ESCALATION": 30,
        },
    )
    assert result.ranked_actions
    assert result.ranked_actions[0].action_id in {
        "STUDY_SCHEDULE",
        "VLE_ENGAGEMENT",
    }
    assert all(
        item.risk_reduction > 0.0 for item in result.ranked_actions
    )
    assert result.rejected_actions[0].action_id == "ADVISOR_ESCALATION"
