from __future__ import annotations

import numpy as np
import pytest

from src.recommend_hybrid.counterfactual.feature_authority import (
    PreprocessedOULADFeatureAuthority,
)
from src.recommend_hybrid.counterfactual.oulad_tensor import BASE_CHANNELS
from src.recommend_hybrid.exceptions import ContractValidationError


class AffinePreprocessor:
    def __init__(self):
        self.mean = np.arange(165, dtype=np.float32)
        self.scale = np.full(165, 2.0, dtype=np.float32)

    def transform_aggregate(self, raw_aggregate):
        return ((raw_aggregate - self.mean) / self.scale).astype(np.float32)

    def inverse_transform_aggregate(self, transformed_aggregate):
        return (transformed_aggregate * self.scale + self.mean).astype(
            np.float32
        )


class FakeFeatureModule:
    BASE_CHANNELS = BASE_CHANNELS

    @staticmethod
    def _dynamic(base_sequence, mask):
        del mask
        result = np.zeros(
            (base_sequence.shape[0], base_sequence.shape[1], 47),
            dtype=np.float32,
        )
        result[:, :, :16] = base_sequence
        return result

    @staticmethod
    def _aggregate(base_sequence, lengths):
        del lengths
        result = np.zeros((base_sequence.shape[0], 161), dtype=np.float32)
        result[:, :16] = base_sequence.sum(axis=1)
        return result


def test_feature_authority_rebuilds_then_applies_frozen_preprocessor():
    preprocessor = AffinePreprocessor()
    authority = PreprocessedOULADFeatureAuthority(
        preprocessor,
        feature_module=FakeFeatureModule,
    )
    base = np.zeros((1, 3, 16), dtype=np.float32)
    base[0, :, 0] = [1.0, 2.0, 3.0]
    raw_baseline = np.arange(165, dtype=np.float32)[None, :] + 10.0
    model_baseline = preprocessor.transform_aggregate(raw_baseline)
    dynamic, transformed = authority.rebuild(
        base,
        np.array([3]),
        np.ones((1, 3), dtype=bool),
        model_baseline,
    )
    recovered = preprocessor.inverse_transform_aggregate(transformed)
    assert dynamic.shape == (1, 3, 47)
    assert transformed.shape == (1, 165)
    assert recovered[0, 0] == pytest.approx(6.0)
    assert np.array_equal(recovered[:, 161:], raw_baseline[:, 161:])


def test_feature_authority_rejects_wrong_preprocessor_space():
    preprocessor = AffinePreprocessor()
    authority = PreprocessedOULADFeatureAuthority(
        preprocessor,
        feature_module=FakeFeatureModule,
    )
    with pytest.raises(ContractValidationError, match="does not match"):
        authority.rebuild(
            np.zeros((1, 2, 16), dtype=np.float32),
            np.array([2]),
            np.ones((1, 2), dtype=bool),
            np.full((1, 165), np.nan, dtype=np.float32),
        )
