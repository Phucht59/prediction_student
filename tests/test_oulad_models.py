from __future__ import annotations

import numpy as np
import torch

from src.studies.oulad.evaluate import binary_metrics, tune_threshold, validate_binary_probabilities
from src.studies.oulad.models_deep import DEEP_CONFIGS, OULADNet


def test_binary_threshold_and_metrics_recompute():
    y = np.array([0, 0, 1, 1])
    probabilities = np.array([0.1, 0.2, 0.8, 0.9])
    validate_binary_probabilities(probabilities)
    threshold, score = tune_threshold(y, probabilities)
    metrics = binary_metrics(y, probabilities, threshold)
    assert score == metrics["macro_f1"] == 1.0
    assert metrics["at_risk_recall"] == 1.0


def test_all_deep_model_shapes_and_padding_contract():
    batch, timesteps, channels, tabular, static = 4, 8, 16, 20, 7
    sequence = torch.randn(batch, timesteps, channels)
    lengths = torch.tensor([8, 7, 5, 3])
    mask = (torch.arange(timesteps)[None, :] < lengths[:, None]).float()
    sequence = sequence * mask.unsqueeze(-1)
    for candidate, config in DEEP_CONFIGS.items():
        model = OULADNet(candidate, channels, tabular, static, config)
        logits = model(sequence, lengths, mask, torch.randn(batch, tabular), torch.randn(batch, static))
        assert logits.shape == (batch,)
        assert torch.isfinite(logits).all()


def test_deep_registry_stays_within_frozen_search_bounds():
    for candidate, config in DEEP_CONFIGS.items():
        assert config["batch_size"] in {64, 128, 256}
        assert 1e-4 <= config["learning_rate"] <= 3e-3
        assert config["max_epochs"] <= 25
        if "kernel_size" in config: assert config["kernel_size"] in {3, 5}
