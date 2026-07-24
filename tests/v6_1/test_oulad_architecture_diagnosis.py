from __future__ import annotations

import copy

import pandas as pd
import pytest
import torch
import yaml

from src.studies.v6.decision_policy import (
    WITHDRAWAL_MECHANISM_STATUS,
    risk_mechanism,
)
from src.studies.v6.recommendation import recommendation_input
from src.studies.v6_1.oulad_architecture import (
    OULADArchitectureDiagnosisNet,
    candidate_specs,
    parameter_breakdown,
)
from src.studies.v6_1.oulad_training import deterministic_seed


@pytest.fixture(scope="module")
def profile():
    return pd.read_parquet("artifacts/v6/prediction/risk_profiles.parquet").iloc[
        0
    ].to_dict()


def _config():
    protocol = yaml.safe_load(
        open("configs/v6_1/oulad_architecture_diagnosis.yaml", encoding="utf-8")
    )
    return {**protocol["current_architecture"], **protocol["training"]}


def _batch():
    deterministic_seed(42)
    lengths = torch.tensor([20, 11, 3, 1])
    mask = (torch.arange(20)[None, :] < lengths[:, None]).float()
    sequence = torch.randn(4, 20, 47) * mask.unsqueeze(-1)
    return sequence, lengths, mask, torch.randn(4, 49), torch.randn(4, 13)


@pytest.mark.parametrize("candidate", list(candidate_specs()))
def test_candidate_shapes_and_finite_outputs(candidate):
    model = OULADArchitectureDiagnosisNet(
        47, 49, 13, _config(), candidate_specs()[candidate]
    )
    logits, diagnostic = model(*_batch(), return_diagnostics=True)
    assert logits.shape == (4,)
    assert torch.isfinite(logits).all()
    assert set(diagnostic) == {"gates", "cnn_norm", "lstm_norm", "expert_cosine"}


@pytest.mark.parametrize(
    "candidate",
    [
        "A1_cnn_small_temporal",
        "A2_bilstm_current_temporal",
        "A4_serial_current_full",
        "D_serial_with_cnn_skip",
        "E_parallel_concat",
    ],
)
def test_padding_values_cannot_change_prediction(candidate):
    deterministic_seed(42)
    model = OULADArchitectureDiagnosisNet(
        47, 49, 13, _config(), candidate_specs()[candidate]
    ).eval()
    first = list(_batch())
    second = [value.clone() for value in first]
    invalid = ~second[2].bool()
    second[0][invalid] = 1_000_000
    with torch.no_grad():
        baseline = model(*first)
        changed = model(*second)
    assert torch.allclose(baseline, changed, atol=1e-6, rtol=0)


def test_current_full_parameter_count_exactly_reproduces_frozen_model():
    model = OULADArchitectureDiagnosisNet(
        47, 49, 13, _config(), candidate_specs()["A4_serial_current_full"]
    )
    assert parameter_breakdown(model)["total"] == 99_443


def test_parameter_matched_cnn_is_within_ten_percent_of_bilstm_encoder():
    config = _config()
    cnn = parameter_breakdown(
        OULADArchitectureDiagnosisNet(
            47, 49, 13, config, candidate_specs()["B2_cnn_matched_temporal"]
        )
    )
    bilstm = parameter_breakdown(
        OULADArchitectureDiagnosisNet(
            47, 49, 13, config, candidate_specs()["A2_bilstm_current_temporal"]
        )
    )
    cnn_encoder = cnn["input_projection"] + cnn["cnn"]
    bilstm_encoder = bilstm["input_projection"] + bilstm["bilstm"]
    assert abs(cnn_encoder - bilstm_encoder) / max(cnn_encoder, bilstm_encoder) <= 0.10


def test_recommendation_observed_state_is_not_synthesized_from_probabilities(profile):
    observed = {
        "activity_level": 0.2,
        "inactivity_streak": 3.0,
        "assessment_progress": 0.4,
        "grade_trend": -0.1,
        "source": "REAL_PRE_CUTOFF_TEST_FIXTURE",
    }
    changed = copy.deepcopy(profile)
    changed["withdrawal_risk_horizon"] = 1 - profile["withdrawal_risk_horizon"]
    changed["probability_fail"] = 0.8
    changed["probability_pass"] = 0.1
    changed["probability_distinction"] = 0.1
    first = recommendation_input(profile, observed_learning_state=observed)
    second = recommendation_input(changed, observed_learning_state=observed)
    assert first["student_learning_state"] == second["student_learning_state"] == observed


def test_missing_real_observed_state_forces_missing_values(profile):
    state = recommendation_input(profile)["student_learning_state"]
    assert all(
        state[name] is None
        for name in (
            "activity_level",
            "inactivity_streak",
            "assessment_progress",
            "grade_trend",
        )
    )
    assert state["source"] == "MISSING_REAL_PRE_CUTOFF_FEATURES"


def test_unreliable_withdrawal_head_cannot_assert_engagement_mechanism(profile):
    changed = copy.deepcopy(profile)
    changed["confidence_level"] = "HIGH_CONFIDENCE"
    changed["decision_status"] = "PREDICTED"
    changed["withdrawal_risk_horizon"] = 0.99
    changed["probability_fail"] = 0.1
    changed["probability_pass"] = 0.8
    changed["probability_distinction"] = 0.1
    assert risk_mechanism(changed) == "GENERAL_RISK"
    assert WITHDRAWAL_MECHANISM_STATUS == "EXPLORATORY_DISABLED_FOR_RECOMMENDATION"
