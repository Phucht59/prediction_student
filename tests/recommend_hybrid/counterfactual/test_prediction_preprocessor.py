from __future__ import annotations

import numpy as np
import pytest
import torch

from src.recommend_hybrid.contracts import CheckpointReference, Stage
from src.recommend_hybrid.exceptions import AuthorityValidationError
from src.recommend_hybrid.prediction_adapter import (
    PARAMETER_COUNT,
    HybridPredictionAdapter,
)


class ParameterCountAuthority(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.authority_parameters = torch.nn.Parameter(
            torch.zeros(PARAMETER_COUNT)
        )


def _reference():
    return CheckpointReference(
        checkpoint_id="test-checkpoint",
        path="artifacts/test-checkpoint.pt",
        sha256="a" * 64,
        fold=0,
        seed=42,
    )


def _adapter_with_full_preprocessor():
    mean = np.arange(165, dtype=np.float64)
    scale = np.linspace(1.0, 3.0, 165, dtype=np.float64)
    return HybridPredictionAdapter(
        (ParameterCountAuthority(),),
        (_reference(),),
        stage=Stage.MIDDLE_50,
        fold=0,
        aggregate_mean=mean,
        aggregate_scale=scale,
        static_num_cols=(
            "num_of_prev_attempts",
            "studied_credits",
            "registration_lead_time",
            "module_presentation_length",
        ),
        static_num_mean=np.array([1.0, 60.0, 20.0, 240.0]),
        static_num_scale=np.array([1.0, 30.0, 10.0, 20.0]),
        static_categories={
            "code_module": ("AAA", "BBB", "CCC"),
            "presentation_season": (
                "B",
                "J",
                "__MISSING__",
                "S1",
                "S2",
                "S3",
            ),
        },
    )


def test_frozen_aggregate_preprocessor_roundtrip_and_hash():
    adapter = _adapter_with_full_preprocessor()
    mean = np.arange(165, dtype=np.float64)
    raw = np.vstack([mean + 1.0, mean + 2.0]).astype(np.float32)
    transformed = adapter.transform_aggregate(raw)
    recovered = adapter.inverse_transform_aggregate(transformed)
    assert transformed.shape == (2, 165)
    assert np.allclose(recovered, raw, rtol=1e-6, atol=1e-5)
    assert adapter.aggregate_preprocessor_hash is not None
    assert adapter.frozen_preprocessor_hash is not None
    assert adapter.frozen_preprocessor_hash == adapter.frozen_preprocessor_hash


def test_frozen_static_preprocessor_matches_training_contract():
    adapter = _adapter_with_full_preprocessor()
    transformed = adapter.transform_static(
        {
            "num_of_prev_attempts": [1, 3],
            "studied_credits": [60, 90],
            "registration_lead_time": [20, None],
            "module_presentation_length": [240, 260],
            "code_module": ["AAA", "CCC"],
            "presentation_season": ["J", None],
        }
    )
    assert transformed.shape == (2, 13)
    assert np.allclose(transformed[0, :4], 0.0)
    assert np.allclose(transformed[1, :4], [2.0, 1.0, -2.0, 1.0])
    # Numeric 4 + code module one-hot 3 + season one-hot 6.
    assert transformed[0, 4:7].tolist() == [1.0, 0.0, 0.0]
    assert transformed[1, 4:7].tolist() == [0.0, 0.0, 1.0]
    assert transformed[0, 7:13].tolist() == [0.0, 1.0, 0.0, 0.0, 0.0, 0.0]
    assert transformed[1, 7:13].tolist() == [0.0, 0.0, 1.0, 0.0, 0.0, 0.0]


def test_adapter_rejects_partial_aggregate_preprocessor():
    with pytest.raises(AuthorityValidationError, match="supplied together"):
        HybridPredictionAdapter(
            (ParameterCountAuthority(),),
            (_reference(),),
            stage=Stage.MIDDLE_50,
            fold=0,
            aggregate_mean=np.zeros(165),
        )


def test_adapter_rejects_partial_static_preprocessor():
    with pytest.raises(AuthorityValidationError, match="supplied together"):
        HybridPredictionAdapter(
            (ParameterCountAuthority(),),
            (_reference(),),
            stage=Stage.MIDDLE_50,
            fold=0,
            static_num_cols=("studied_credits",),
        )
