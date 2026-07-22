from __future__ import annotations

import numpy as np
import torch

from src.studies.v5_1.oulad.pretraining import (
    MaskedWeekReconstructorV51,
    deterministic_week_mask,
    masked_reconstruction_loss,
    reconstruction_channel_indices,
    temporal_state_dict,
)


CHANNEL_ORDER = (
    "total_clicks",
    "active_days",
    "unique_sites",
    "unique_activity_types",
    "content_clicks",
    "forum_clicks",
    "quiz_clicks",
    "assessment_related_clicks",
    "submitted_assessment_count",
    "late_submission_count",
    "available_score_count",
    "cumulative_mean_score",
    "cumulative_weighted_score",
    "days_since_last_vle_activity",
    "weeks_without_activity",
    "score_missing_mask",
    *tuple(f"derived_{index}" for index in range(31)),
)


def _config() -> dict[str, object]:
    return {
        "input_projection": 12,
        "conv_channels": 8,
        "kernels": [2, 3],
        "dilation": 1,
        "lstm_hidden": 10,
        "lstm_layers": 1,
        "pooling": "masked_mean_max",
        "pooling_projection": 12,
        "dropout": 0.0,
    }


def test_masked_week_selection_never_masks_padding_or_entire_sequence() -> None:
    valid = np.array(
        [[1, 1, 1, 1, 0], [1, 1, 0, 0, 0], [1, 0, 0, 0, 0]], dtype=bool
    )
    selected = deterministic_week_mask(valid, 0.5, 42)
    assert not (selected & ~valid).any()
    assert np.all(selected.sum(axis=1) < valid.sum(axis=1))
    assert selected[2].sum() == 0


def test_reconstruction_loss_uses_only_masked_weeks_and_registered_channels() -> None:
    indices = reconstruction_channel_indices(CHANNEL_ORDER)
    target = torch.randn(2, 4, 47)
    prediction = target[:, :, list(indices)].clone()
    week_mask = torch.tensor([[False, True, False, False], [False, False, True, False]])
    assert float(masked_reconstruction_loss(prediction, target, week_mask, indices)) == 0.0
    prediction[~week_mask] = 1000.0
    assert float(masked_reconstruction_loss(prediction, target, week_mask, indices)) == 0.0


def test_pretraining_model_and_temporal_transfer_state_have_expected_shapes() -> None:
    indices = reconstruction_channel_indices(CHANNEL_ORDER)
    model = MaskedWeekReconstructorV51(47, indices, _config())
    sequence = torch.randn(3, 6, 47)
    lengths = torch.tensor([6, 4, 2])
    mask = (torch.arange(6)[None, :] < lengths[:, None]).float()
    prediction = model(sequence * mask.unsqueeze(-1), lengths, mask)
    assert prediction.shape == (3, 6, len(indices))
    state = temporal_state_dict(model.state_dict())
    assert state
    assert all(name.startswith("temporal.") for name in state)
    assert not any("reconstruction_head" in name for name in state)
