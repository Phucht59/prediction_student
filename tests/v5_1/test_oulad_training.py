from __future__ import annotations

import numpy as np

from src.studies.v5_1.oulad.data import OULADInputsV51, OULADPreprocessorsV51
from src.studies.v5_1.oulad.training import (
    choose_threshold,
    fit_prepared_oulad_model,
    transform_temporal_order,
)


def _inputs(records: int, seed: int) -> OULADInputsV51:
    rng = np.random.default_rng(seed)
    weeks = 6
    lengths = rng.integers(2, weeks + 1, size=records, dtype=np.int64)
    mask = (np.arange(weeks)[None, :] < lengths[:, None]).astype(np.float32)
    sequence = rng.normal(size=(records, weeks, 47)).astype(np.float32) * mask[..., None]
    return OULADInputsV51(
        sequence=sequence,
        lengths=lengths,
        mask=mask,
        aggregate=rng.normal(size=(records, 12)).astype(np.float32),
        static=rng.normal(size=(records, 5)).astype(np.float32),
        target=np.tile([0.0, 1.0], records // 2 + 1)[:records].astype(np.float32),
        preprocessors=OULADPreprocessorsV51(),
        aggregate_columns=tuple(f"feature_{index}" for index in range(12)),
    )


def _config() -> dict[str, object]:
    return {
        "input_projection": 12,
        "conv_channels": 8,
        "kernels": [2, 3],
        "dilation": 1,
        "lstm_hidden": 10,
        "lstm_layers": 1,
        "pooling": "masked_attention",
        "pooling_projection": 12,
        "aggregate_hidden": 12,
        "static_hidden": 8,
        "fusion_hidden": 12,
        "fusion": "gated_residual",
        "branch_dropout": 0.0,
        "dropout": 0.0,
        "loss": "standard_bce",
        "learning_rate": 0.001,
        "weight_decay": 0.0,
        "batch_size": 8,
        "gradient_clip": 1.0,
        "parameter_limit": 1_500_000,
    }


def test_temporal_order_transforms_only_valid_weeks() -> None:
    inputs = _inputs(8, 1)
    reversed_inputs = transform_temporal_order(inputs, "reversed")
    for row, length in enumerate(inputs.lengths):
        np.testing.assert_array_equal(
            reversed_inputs.sequence[row, :length], inputs.sequence[row, :length][::-1]
        )
        assert not reversed_inputs.sequence[row, length:].any()
    shuffled_a = transform_temporal_order(inputs, "shuffled", 42)
    shuffled_b = transform_temporal_order(inputs, "shuffled", 42)
    np.testing.assert_array_equal(shuffled_a.sequence, shuffled_b.sequence)


def test_prepared_training_replays_and_reports_diagnostics() -> None:
    fit = fit_prepared_oulad_model(
        _inputs(24, 2),
        _inputs(12, 3),
        config=_config(),
        seed=42,
        fixed_epochs=1,
        device_name="cpu",
    )
    assert fit.probability.shape == (12,)
    assert fit.replay_max_abs_difference == 0.0
    assert fit.attention_padding_max == 0.0
    assert fit.attention_entropy_mean is not None
    assert fit.gate_statistics["per_branch_mean"]


def test_threshold_is_selected_from_registered_grid() -> None:
    result = choose_threshold(np.array([0, 0, 1, 1]), np.array([0.1, 0.4, 0.6, 0.9]))
    assert 0.2 <= result["threshold"] <= 0.8
    assert result["inner_macro_f1"] == 1.0
