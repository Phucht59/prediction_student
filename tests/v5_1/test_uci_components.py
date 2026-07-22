from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import torch

from src.studies.v5_1.common.uci_data import (
    PRIMARY_CONTEXT_FEATURES,
    SENSITIVITY_CONTEXT_FEATURES,
    TEMPORAL_CHANNELS,
    build_temporal_features,
)
from src.studies.v5_1.common.uci_model import UCIHybridV51, count_parameters, gate_statistics


def _config(fusion: str) -> dict[str, object]:
    return {
        "input_projection": 16,
        "cnn_channels": 8,
        "lstm_hidden": 16,
        "lstm_layers": 1,
        "context_hidden": 16,
        "context_layers": 1,
        "fusion_hidden": 16,
        "fusion": fusion,
        "dropout": 0.1,
        "activation": "gelu",
    }


def test_temporal_features_are_deterministic_and_do_not_use_g3() -> None:
    frame = pd.DataFrame({"G1": [8, 12, 18], "G2": [10, 9, 18], "G3": [1, 2, 3]})
    changed = frame.assign(G3=[20, 20, 20])
    first = build_temporal_features(frame)
    second = build_temporal_features(changed)
    np.testing.assert_array_equal(first, second)
    assert first.shape == (3, 2, len(TEMPORAL_CHANNELS))
    np.testing.assert_array_equal(first[:, 0, 2:4], 0.0)
    np.testing.assert_array_equal(first[:, 1, 6], np.array([1.0, -1.0, 0.0], dtype=np.float32))


def test_primary_context_excludes_absences_and_sensitivity_is_explicit() -> None:
    assert "absences" not in PRIMARY_CONTEXT_FEATURES
    assert "absences" in SENSITIVITY_CONTEXT_FEATURES
    assert set(SENSITIVITY_CONTEXT_FEATURES) - set(PRIMARY_CONTEXT_FEATURES) == {"absences"}


@pytest.mark.parametrize("fusion", ["concatenation", "gated", "film_residual"])
def test_uci_hybrid_dimensions_and_heads(fusion: str) -> None:
    model = UCIHybridV51(len(TEMPORAL_CHANNELS), 20, _config(fusion))
    output = model(torch.randn(5, 2, len(TEMPORAL_CHANNELS)), torch.randn(5, 20))
    assert output["classification"].shape == (5, 3)
    assert output["regression"].shape == (5,)
    assert output["ordinal"].shape == (5, 2)
    assert output["temporal_norm"].shape == (5,)
    assert output["context_norm"].shape == (5,)
    assert ("gate" in output) is (fusion != "concatenation")
    assert ("film_gamma" in output) is (fusion == "film_residual")
    assert count_parameters(model) < 1_500_000


def test_even_kernel_preserves_two_timesteps_and_residual_receives_gradient() -> None:
    model = UCIHybridV51(len(TEMPORAL_CHANNELS), 12, _config("gated"))
    output = model(torch.randn(4, 2, len(TEMPORAL_CHANNELS)), torch.randn(4, 12))
    output["classification"].sum().backward()
    assert model.temporal.convolutions[1].weight.grad is not None
    assert model.temporal.residual.weight.grad is not None


def test_gate_collapse_diagnostic() -> None:
    healthy = gate_statistics(torch.tensor([[0.25, 0.75], [0.60, 0.40]]))
    collapsed = gate_statistics(torch.full((8, 4), 0.9999))
    absent = gate_statistics(None)
    assert healthy["collapsed"] is False
    assert collapsed["collapsed"] is True
    assert absent["mean"] is None


def test_temporal_contract_rejects_missing_or_out_of_range_grades() -> None:
    with pytest.raises(ValueError, match="Missing temporal"):
        build_temporal_features(pd.DataFrame({"G1": [10]}))
    with pytest.raises(ValueError, match="inside 0..20"):
        build_temporal_features(pd.DataFrame({"G1": [10], "G2": [21]}))


@pytest.mark.parametrize("variant", ["cnn_bilstm", "cnn_only", "bilstm_only"])
def test_registered_uci_temporal_ablation_shapes(variant: str) -> None:
    model = UCIHybridV51(
        len(TEMPORAL_CHANNELS), 12, {**_config("gated"), "temporal_variant": variant}
    )
    output = model(torch.randn(4, 2, len(TEMPORAL_CHANNELS)), torch.randn(4, 12))
    assert output["classification"].shape == (4, 3)
