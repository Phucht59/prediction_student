from __future__ import annotations

import numpy as np
import torch

from src.studies.v5_1.common.uci_model import UCIHybridV51
from src.studies.v5_1.common.uci_training import (
    UCIInputsV51,
    encoder_parameter_names,
    fit_uci_model_v5_1,
    multitask_loss,
    resample_deep_training,
    set_encoder_trainable,
)
from src.studies.v5_1.common.uci_transfer import (
    SharedTrunkSubjectHeadsV51,
    combine_subject_inputs,
    fit_shared_subject_model,
    overlap_safe_source_indices,
)


def _config(**updates: object) -> dict[str, object]:
    config: dict[str, object] = {
        "input_projection": 8,
        "cnn_channels": 4,
        "lstm_hidden": 6,
        "lstm_layers": 1,
        "context_hidden": 8,
        "context_layers": 1,
        "fusion_hidden": 8,
        "fusion": "gated",
        "dropout": 0.0,
        "activation": "gelu",
        "objective": "classification_plus_huber_regression_plus_ordinal",
        "regression_weight": 0.1,
        "ordinal_weight": 0.05,
        "learning_rate": 0.001,
        "weight_decay": 0.0,
        "batch_size": 8,
        "gradient_clip": 1.0,
        "parameter_limit": 1_500_000,
        "subject_embedding_dim": 4,
    }
    config.update(updates)
    return config


def _inputs(records: int = 24, context_dim: int = 10, seed: int = 7) -> UCIInputsV51:
    rng = np.random.default_rng(seed)
    target = np.tile(np.arange(3), records // 3 + 1)[:records].astype(np.int64)
    return UCIInputsV51(
        temporal=rng.normal(size=(records, 2, 7)).astype(np.float32),
        context=rng.normal(size=(records, context_dim)).astype(np.float32),
        target=target,
        raw_g3=(target * 5 + 6).astype(np.float32),
    )


def test_multitask_loss_respects_registered_weights() -> None:
    output = {
        "classification": torch.tensor([[2.0, 0.0, -1.0], [0.0, 2.0, -1.0]]),
        "regression": torch.tensor([0.0, 0.0]),
        "ordinal": torch.zeros(2, 2),
    }
    total, parts = multitask_loss(
        output,
        torch.tensor([0, 1]),
        torch.tensor([5.0, 10.0]),
        config=_config(),
        class_weights=None,
        regression_mean=7.5,
        regression_std=2.5,
    )
    expected = parts["classification_loss"] + 0.1 * parts["regression_loss"] + 0.05 * parts["ordinal_loss"]
    torch.testing.assert_close(total, expected)


def test_deep_random_duplication_preserves_real_rows() -> None:
    inputs = _inputs(12)
    inputs = UCIInputsV51(inputs.temporal, inputs.context, np.array([0] * 8 + [1] * 3 + [2]), inputs.raw_g3)
    sampled, before, after = resample_deep_training(inputs, "random_sample_duplication", 42)
    assert before == {0: 8, 1: 3, 2: 1}
    assert after == {0: 8, 1: 8, 2: 8}
    original = {row.tobytes() for row in inputs.temporal}
    assert all(row.tobytes() in original for row in sampled.temporal)


def test_encoder_freeze_switch_covers_only_registered_encoder_parameters() -> None:
    model = UCIHybridV51(7, 10, _config())
    encoder_names = set(encoder_parameter_names(model))
    set_encoder_trainable(model, False)
    assert encoder_names
    assert all(not parameter.requires_grad for name, parameter in model.named_parameters() if name in encoder_names)
    assert all(parameter.requires_grad for name, parameter in model.named_parameters() if name not in encoder_names)
    set_encoder_trainable(model, True)
    assert all(parameter.requires_grad for parameter in model.parameters())


def test_overlap_filter_removes_source_groups_seen_in_target_validation() -> None:
    source = np.array(["a", "b", "c", "b"])
    selected = overlap_safe_source_indices(source, np.array(["b", "z"]))
    np.testing.assert_array_equal(selected, np.array([0, 2]))


def test_shared_trunk_dispatches_to_subject_specific_heads() -> None:
    model = SharedTrunkSubjectHeadsV51(7, 10, _config())
    temporal = torch.randn(4, 2, 7)
    context = torch.randn(4, 10)
    subject = torch.tensor([0, 1, 0, 1])
    output = model(temporal, context, subject)
    assert output["classification"].shape == (4, 3)
    assert output["regression"].shape == (4,)
    assert output["ordinal"].shape == (4, 2)
    assert model.classifiers[0] is not model.classifiers[1]


def test_freeze_unfreeze_training_and_checkpoint_replay_smoke() -> None:
    train = _inputs(24, seed=1)
    evaluation = _inputs(12, seed=2)
    fit = fit_uci_model_v5_1(
        train,
        evaluation,
        config=_config(),
        seed=42,
        fixed_epochs=2,
        device_name="cpu",
        freeze_epochs=1,
    )
    assert [row["encoder_frozen"] for row in fit.history] == [True, False]
    assert fit.probability.shape == (12, 3)
    assert fit.regression.shape == (12,)
    assert fit.ordinal_probability.shape == (12, 2)
    assert fit.replay_max_abs_difference == 0.0


def test_shared_subject_training_checkpoint_replay_smoke() -> None:
    mat, por = _inputs(12, seed=3), _inputs(15, seed=4)
    combined = combine_subject_inputs(mat, por)
    evaluation_base = _inputs(9, seed=5)
    evaluation = combine_subject_inputs(evaluation_base, _inputs(3, seed=6))
    fit = fit_shared_subject_model(
        combined,
        evaluation,
        config=_config(objective="classification_only"),
        seed=42,
        fixed_epochs=1,
        device_name="cpu",
    )
    assert fit.probability.shape == (12, 3)
    assert fit.replay_max_abs_difference == 0.0
