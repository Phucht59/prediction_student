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


def test_frozen_aggregate_preprocessor_roundtrip_and_hash():
    mean = np.arange(165, dtype=np.float64)
    scale = np.linspace(1.0, 3.0, 165, dtype=np.float64)
    adapter = HybridPredictionAdapter(
        (ParameterCountAuthority(),),
        (_reference(),),
        stage=Stage.MIDDLE_50,
        fold=0,
        aggregate_mean=mean,
        aggregate_scale=scale,
    )
    raw = np.vstack([mean + 1.0, mean + 2.0]).astype(np.float32)
    transformed = adapter.transform_aggregate(raw)
    recovered = adapter.inverse_transform_aggregate(transformed)
    assert transformed.shape == (2, 165)
    assert np.allclose(recovered, raw, rtol=1e-6, atol=1e-5)
    assert adapter.aggregate_preprocessor_hash is not None
    assert adapter.aggregate_preprocessor_hash == adapter.aggregate_preprocessor_hash


def test_adapter_rejects_partial_aggregate_preprocessor():
    with pytest.raises(AuthorityValidationError, match="supplied together"):
        HybridPredictionAdapter(
            (ParameterCountAuthority(),),
            (_reference(),),
            stage=Stage.MIDDLE_50,
            fold=0,
            aggregate_mean=np.zeros(165),
        )
