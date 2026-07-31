from __future__ import annotations

import inspect

import numpy as np
import torch

from src.pipelines import oulad
from src.training import phase9_endpoint_recovery as phase9


def test_phase9_architecture_is_frozen() -> None:
    identity = phase9.architecture_identity()
    assert identity["architecture_id"] == "H1_TABULAR_RESIDUAL_EXPERT"
    assert identity["parameter_count"] == 160_492


def test_score_proxy_is_rejected_without_release_timestamp() -> None:
    authority = phase9.score_feature_authority()
    assert authority["decision"] == "SCORE_PROXY_REJECTED"
    assert authority["score_release_timestamp_present_in_raw_oulad"] is False
    assert authority["score_features_used_in_h1r"] is False


def test_h0_compact_mapping_keeps_h1_input_dimension_and_zeroes_score() -> None:
    sequence = np.ones((2, 4, len(oulad.CHANNELS)), dtype=np.float32)
    mask = np.ones((2, 4), dtype=np.float32)
    aggregate = phase9.valid_h0_aggregate(sequence, mask)
    assert aggregate.shape == (2, 165)
    assert np.all(aggregate[:, 49:] == 0)
    # Historical score-derived compact positions are forced to zero.
    score_positions = list(range(32, 40)) + [46, 47]
    assert np.all(aggregate[:, score_positions] == 0)


def test_phase9_selection_api_has_no_outer_label_argument() -> None:
    signature = inspect.signature(phase9.evaluate_candidate)
    assert "outer_y_test" not in signature.parameters
    assert "outer_labels" not in signature.parameters


def test_holdout_audit_does_not_rebrand_observed_data() -> None:
    audit = phase9.holdout_availability_audit()
    assert audit["new_untouched_holdout_available"] is False
    assert audit["random_resplit_is_new_holdout"] is False
    assert audit["confirmation_allowed"] is False


def test_phase9_budget_and_architecture_contract() -> None:
    assert phase9.MAX_OPTUNA_TRIALS == 24
    source = inspect.getsource(phase9.run_supervisor)
    assert "H1_TABULAR_RESIDUAL_EXPERT" not in source  # fixed via module constant
    assert "outer_labels_used\": False" in source
    assert "selected_frozen_configs" in source


def test_h1_forward_loss_backward_and_branch_connections() -> None:
    config = phase9.H0_CONFIG
    model = phase9.make_model("H1_TABULAR_RESIDUAL_EXPERT", 165, 13, config)
    sequence = torch.randn(4, 6, 47)
    lengths = torch.tensor([6, 5, 4, 3])
    mask = torch.arange(6)[None, :] < lengths[:, None]
    aggregate = torch.randn(4, 165)
    static = torch.randn(4, 13)
    output = model(sequence, lengths, mask.float(), aggregate, static)
    loss = torch.nn.functional.binary_cross_entropy_with_logits(
        output["binary_logit"], torch.tensor([0.0, 1.0, 0.0, 1.0])
    )
    loss.backward()
    assert model.backbone.temporal.input_projection.weight.grad is not None
    assert model.tabular_risk_head.weight.grad is not None


def test_future_mask_perturbation_does_not_change_valid_h0_aggregate() -> None:
    rng = np.random.default_rng(9)
    sequence = rng.normal(size=(3, 6, len(oulad.CHANNELS))).astype(np.float32)
    mask = np.asarray(
        [[1, 1, 1, 0, 0, 0], [1, 1, 1, 1, 0, 0], [1, 1, 0, 0, 0, 0]],
        dtype=np.float32,
    )
    changed = sequence.copy()
    changed[mask == 0] = 1_000_000
    np.testing.assert_allclose(
        phase9.valid_h0_aggregate(sequence, mask),
        phase9.valid_h0_aggregate(changed, mask),
        rtol=0,
        atol=0,
    )
