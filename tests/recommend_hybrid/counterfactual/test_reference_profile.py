from __future__ import annotations

import inspect

import numpy as np
import pytest

from src.recommend_hybrid.counterfactual.oulad_tensor import BASE_CHANNELS
from src.recommend_hybrid.counterfactual.reference_profile import (
    OULADReferenceProfileBuilder,
    REFERENCE_SPECS,
)
from src.recommend_hybrid.exceptions import ContractValidationError


def _training_sequence():
    sequence = np.zeros((3, 5, 47), dtype=np.float32)
    index = {name: i for i, name in enumerate(BASE_CHANNELS)}
    lengths = np.array([2, 4, 3], dtype=int)
    observed = {
        "total_clicks": [10, 20, 30, 40, 50, 60, 70, 80, 90],
        "active_days": [1, 2, 3, 4, 5, 6, 7, 2, 3],
        "content_clicks": [5, 10, 15, 20, 25, 30, 35, 40, 45],
        "unique_sites": [1, 2, 3, 4, 5, 6, 7, 3, 2],
        "quiz_clicks": [1, 2, 3, 4, 5, 6, 7, 8, 9],
        "assessment_related_clicks": [2, 4, 6, 8, 10, 12, 14, 16, 18],
    }
    positions = [
        (0, 0),
        (0, 1),
        (1, 0),
        (1, 1),
        (1, 2),
        (1, 3),
        (2, 0),
        (2, 1),
        (2, 2),
    ]
    for channel, values in observed.items():
        for (row, week), value in zip(positions, values, strict=True):
            sequence[row, week, index[channel]] = value
    # Padded values must never enter a training reference.
    for row, length in enumerate(lengths):
        sequence[row, length:, :16] = 9999.0
    return sequence, lengths


def test_reference_profile_ignores_padded_weeks_and_is_deterministic():
    sequence, lengths = _training_sequence()
    builder = OULADReferenceProfileBuilder(minimum_positive_observations=3)
    kwargs = dict(
        sequence=sequence,
        lengths=lengths,
        fold=1,
        stage="M1_MIDDLE_50PCT",
        course_key="AAA-2014J",
    )
    first = builder.build(**kwargs)
    second = builder.build(**kwargs)
    assert first.to_dict() == second.to_dict()
    assert first.observed_week_count == 9
    assert max(first.values().values()) < 9999.0


def test_reference_profile_contains_exact_tensor_reference_keys():
    sequence, lengths = _training_sequence()
    profile = OULADReferenceProfileBuilder(
        minimum_positive_observations=3
    ).build(
        sequence=sequence,
        lengths=lengths,
        fold=0,
        stage="E1_EARLY_20PCT",
        course_key="BBB-2013B",
    )
    assert list(profile.values()) == [item[0] for item in REFERENCE_SPECS]
    assert profile.sample_role == "TRAIN"
    assert profile.reference_scope == "TRAINING_FOLD_COURSE_STAGE_ONLY"
    assert profile.profile_id.startswith("oulad_ref_")


def test_reference_builder_rejects_non_training_samples():
    sequence, lengths = _training_sequence()
    with pytest.raises(ContractValidationError, match="validation/test"):
        OULADReferenceProfileBuilder().build(
            sequence=sequence,
            lengths=lengths,
            fold=0,
            stage="L1_LATE_75PCT",
            course_key="CCC-2014B",
            sample_role="TEST",
        )


def test_reference_builder_has_no_target_or_label_input():
    parameters = inspect.signature(OULADReferenceProfileBuilder.build).parameters
    assert "target" not in parameters
    assert "targets" not in parameters
    assert "label" not in parameters
    assert "labels" not in parameters
