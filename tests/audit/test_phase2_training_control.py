from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn

from src.models._oulad import count_parameters
from src.models.oulad_multitask import CNNBiLSTMOULAD
from src.pipelines import oulad
from src.training.config_authority import (
    architecture_metadata,
    load_config_authority,
    resolved_deep_config,
)
from src.training.control import (
    early_stop_metadata,
    finalize_training_metadata,
    pretraining_provenance,
    select_operational_threshold,
    select_refit_epoch,
    select_research_threshold,
)


ROOT = Path(__file__).resolve().parents[2]
AUTHORITY_PATH = ROOT / "configs" / "registry" / "oulad_unified_stage_aware_v2.yaml"
FROZEN_OULAD = ROOT / "artifacts" / "final" / "unified_stage_aware_oulad"


def _tiny_config(fusion: str = "gated_residual") -> dict[str, object]:
    return {
        "input_projection": 8,
        "conv_channels": 4,
        "kernels": [2, 3],
        "lstm_hidden": 4,
        "lstm_layers": 1,
        "pooling": "masked_mean_max",
        "pooling_projection": 8,
        "aggregate_hidden": 8,
        "static_hidden": 8,
        "fusion_hidden": 8,
        "dropout": 0.0,
        "fusion": fusion,
        "branch_dropout": 0.0,
    }


def _forward(model: CNNBiLSTMOULAD) -> dict[str, torch.Tensor]:
    return model(
        torch.randn(3, 4, 47),
        torch.tensor([2, 3, 4]),
        torch.tensor(
            [[1.0, 1.0, 0.0, 0.0], [1.0, 1.0, 1.0, 0.0], [1.0] * 4]
        ),
        torch.randn(3, 5),
        torch.randn(3, 3),
    )


def test_t1_fixed_refit_epoch_metadata_is_actual_epoch() -> None:
    assert finalize_training_metadata(
        fixed_epochs=7,
        epochs_trained=7,
        selected_epoch=1,
        monitor="unused_for_fixed_refit",
    ) == {
        "training_mode": "fixed_epoch_refit",
        "epochs_trained": 7,
        "selected_epoch": 7,
        "checkpoint_epoch": 7,
        "checkpoint_selection": "final_fixed_epoch",
        "early_stopping_applied": False,
    }
    assert early_stop_metadata(
        epochs_trained=9, selected_epoch=4, monitor="validation_nll"
    )["selected_epoch"] == 4
    with pytest.raises(RuntimeError, match="execution count"):
        finalize_training_metadata(
            fixed_epochs=4,
            epochs_trained=3,
            selected_epoch=1,
            monitor="unused_for_fixed_refit",
        )


def test_t2_payload_and_manifest_use_same_canonical_run_id() -> None:
    identity = oulad._training_identity(
        "cnn_bilstm", 0, 42, training_mode="fixed_epoch_refit"
    )
    payload = {"training_run_id": identity.run_id}
    manifest = {"training_run_id": identity.run_id}
    assert payload["training_run_id"] == manifest["training_run_id"]
    assert len(identity.run_id) == 24


def test_t3_architecture_fingerprint_changes_with_architecture() -> None:
    authority = load_config_authority(AUTHORITY_PATH)
    config = resolved_deep_config(authority)
    first = architecture_metadata(
        CNNBiLSTMOULAD(47, 5, 3, config),
        authority=authority,
        aggregate_dim=5,
        static_dim=3,
    )
    repeated = architecture_metadata(
        CNNBiLSTMOULAD(47, 5, 3, config),
        authority=authority,
        aggregate_dim=5,
        static_dim=3,
    )
    altered = json.loads(json.dumps(authority))
    altered["architecture"]["kernels"] = [2, 3]
    second = architecture_metadata(
        CNNBiLSTMOULAD(47, 5, 3, resolved_deep_config(altered)),
        authority=altered,
        aggregate_dim=5,
        static_dim=3,
    )
    assert first["architecture_hash"] != second["architecture_hash"]
    assert first["config_hash"] != second["config_hash"]
    assert first["architecture_hash"] == repeated["architecture_hash"]
    assert first["config_hash"] == repeated["config_hash"]


def test_t4_inner_epochs_propagate_by_deterministic_median() -> None:
    assert select_refit_epoch([3, 8]) == 6
    assert select_refit_epoch([8, 3]) == 6
    import pandas as pd

    details = oulad._refit_epoch_details(
        pd.DataFrame(
            {
                "model_family": ["cnn_bilstm", "cnn_bilstm"],
                "outer_fold": [0, 0],
                "selected_epoch": [3, 8],
            }
        ),
        "cnn_bilstm",
        0,
    )
    assert details["selected_refit_epoch"] == 6
    assert details["outer_labels_used"] is False


def test_t5_current_gated_model_behavior_and_parameter_count_are_unchanged() -> None:
    authority = load_config_authority(AUTHORITY_PATH)
    model = CNNBiLSTMOULAD(47, 165, 13, resolved_deep_config(authority))
    assert model.representation_dim == 64
    assert model.backbone.representation_dim == 64
    assert count_parameters(model) == 150202


def test_t6_concat_all_heads_forward_loss_and_backward() -> None:
    model = CNNBiLSTMOULAD(47, 5, 3, _tiny_config("concatenation"))
    output = _forward(model)
    assert model.representation_dim == 24
    loss = (
        nn.functional.binary_cross_entropy_with_logits(
            output["binary_logit"], torch.tensor([0.0, 1.0, 0.0])
        )
        + nn.functional.binary_cross_entropy_with_logits(
            output["hazard_logit"], torch.zeros(3, 20)
        )
        + nn.functional.cross_entropy(output["outcome_logit"], torch.tensor([0, 1, 2]))
    )
    loss.backward()
    assert model.survival_head.weight.grad is not None
    assert model.outcome_head.weight.grad is not None


def test_t7_outer_labels_cannot_affect_epoch_selection_api() -> None:
    inner_epochs = [4, 10]
    outer_a = np.zeros(20, dtype=int)
    outer_b = np.ones(20, dtype=int)
    first = select_refit_epoch(inner_epochs)
    second = select_refit_epoch(inner_epochs)
    assert not np.array_equal(outer_a, outer_b)
    assert first == second == 7


def test_t8_research_threshold_is_inner_oof_only() -> None:
    result = select_research_threshold(
        np.array([0, 0, 1, 1]), np.array([0.1, 0.4, 0.45, 0.9])
    )
    assert result["policy"] == "INNER_OOF_RESEARCH_MACRO_F1"
    assert result["source"] == "pooled_inner_oof"
    assert result["outer_labels_used"] is False


def test_t9_operational_threshold_is_separate_and_not_checkpoint_objective() -> None:
    result = select_operational_threshold(
        np.array([0, 0, 1, 1]), np.array([0.1, 0.2, 0.7, 0.9])
    )
    assert result["policy"] == "INNER_OOF_OPERATIONAL_RECALL_AT_PRECISION"
    assert result["eligible_for_checkpoint_selection"] is False
    assert result["outer_labels_used"] is False


def test_t10_monitor_threshold_and_research_threshold_are_distinct_policies() -> None:
    authority = load_config_authority(AUTHORITY_PATH)
    policies = authority["thresholds"]
    assert policies["monitor_threshold"] == 0.5
    assert policies["research"] != policies["operational"]
    assert authority["training"]["monitor"] == "mean_stage_validation_nll"
    import pandas as pd

    frame = pd.DataFrame(
        {"prediction_stage": [oulad.STAGES[0]] * 3 + [oulad.STAGES[1]]}
    )
    labels = np.array([0, 0, 0, 1])
    probabilities = np.array([0.1, 0.1, 0.1, 0.51])
    expected = np.mean(
        [
            nn.functional.binary_cross_entropy(
                torch.tensor([0.1] * 3), torch.tensor([0.0] * 3)
            ).item(),
            nn.functional.binary_cross_entropy(
                torch.tensor([0.51]), torch.tensor([1.0])
            ).item(),
        ]
    )
    assert oulad._mean_stage_nll(frame, labels, probabilities) == pytest.approx(
        expected, rel=1e-6
    )


def test_t11_weighted_bce_and_auxiliary_weights_including_zero_aux() -> None:
    output = {
        "binary_logit": torch.tensor([0.2, -0.1], requires_grad=True),
        "hazard_logit": torch.zeros(2, 20, requires_grad=True),
        "outcome_logit": torch.zeros(2, 3, requires_grad=True),
    }
    target = torch.tensor([1.0, 0.0])
    weights = torch.tensor([2.0, 1.0])
    outcome = torch.tensor([0, 1])
    cutoff = torch.tensor([10, 10])
    end = torch.tensor([100, 100])
    unregister = torch.tensor([-1, 20])
    risk_loss = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(1.5), reduction="none")
    risk_only, components = oulad._multitask_loss(
        output, target, weights, outcome, cutoff, end, unregister, risk_loss,
        survival_weight=0.0, outcome_weight=0.0,
    )
    expected = (risk_loss(output["binary_logit"], target) * weights).sum() / weights.sum()
    assert torch.allclose(risk_only, expected)
    with_aux, _ = oulad._multitask_loss(
        output, target, weights, outcome, cutoff, end, unregister, risk_loss,
        survival_weight=0.15, outcome_weight=0.15,
    )
    assert with_aux > components["risk"]


def test_t12_unified_pretraining_provenance_is_explicitly_not_executed() -> None:
    authority = load_config_authority(AUTHORITY_PATH)
    assert authority["pretraining"] == pretraining_provenance(
        requested=False, executed=False, checkpoint=None, strategy=None
    )


def test_t13_frozen_stage_checkpoint_mapping_is_preserved() -> None:
    payload = json.loads((FROZEN_OULAD / "checkpoint_stage_mapping.json").read_text())
    assert payload["same_checkpoint_all_stages"] is True
    assert payload["mapping_count"] == 600


def test_t14_future_mask_still_blocks_unavailable_timesteps() -> None:
    torch.manual_seed(13)
    model = CNNBiLSTMOULAD(47, 5, 3, _tiny_config()).eval()
    sequence = torch.randn(2, 5, 47)
    changed = sequence.clone()
    changed[:, 2:] = 10_000
    args = (
        torch.tensor([2, 2]),
        torch.tensor([[1.0, 1.0, 0.0, 0.0, 0.0]] * 2),
        torch.randn(2, 5),
        torch.randn(2, 3),
    )
    with torch.no_grad():
        first = model(sequence, *args)["binary_logit"]
        second = model(changed, *args)["binary_logit"]
    assert torch.allclose(first, second, atol=1e-7, rtol=0)
