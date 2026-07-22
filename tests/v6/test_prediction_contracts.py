from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from src.studies.v5_1.oulad.runner import _load
from src.studies.v6.contract import ARTIFACT_ROOT, protected_hash_status
from src.studies.v6.multitask import build_temporal_targets


@pytest.fixture(scope="module")
def data():
    return _load()[2]


def test_v5_1_checkpoint_replay_passes():
    value = json.loads(
        (ARTIFACT_ROOT / "prediction/v5_1_reproduction.json").read_text(encoding="utf-8")
    )
    assert value["status"] == "PASS"
    assert max(
        row["probability_max_abs_difference"] for row in value["checkpoint_replays"]
    ) <= 1e-6


def test_same_cohort_split_and_features():
    value = json.loads(
        (ARTIFACT_ROOT / "prediction/v5_1_reproduction.json").read_text(encoding="utf-8")
    )
    assert value["same_cohort"] and value["same_split"] and value["same_feature_order"]


def test_padding_is_zero(data):
    assert np.allclose(data.dynamic_sequence[~data.base.padding_mask], 0.0)


def test_temporal_length_matches_mask(data):
    assert np.array_equal(data.base.padding_mask.sum(axis=1), data.base.valid_lengths)


def test_no_outer_student_overlap(data):
    for fold in range(3):
        train, test = data.v2.outer_indices(fold)
        assert set(data.groups[train]).isdisjoint(set(data.groups[test]))


def test_no_future_role(data):
    assert set(data.development_manifest.role) == {"historical_development"}


def test_probability_seed_and_fold_coverage():
    frame = pd.read_parquet(ARTIFACT_ROOT / "prediction/final/seed_predictions.parquet")
    assert sorted(frame.seed.unique()) == [42, 1201, 2026, 3407, 7319]
    assert sorted(frame.outer_fold.unique()) == [0, 1, 2]
    assert np.isfinite(frame.probability).all()
    assert frame.probability.between(0, 1).all()


def test_calibration_is_train_only():
    value = json.loads(
        (ARTIFACT_ROOT / "prediction/calibration.json").read_text(encoding="utf-8")
    )
    assert value["fit_scope"].endswith("inner_oof_only")
    assert value["outer_test_used_to_fit"] is False


def test_survival_censoring_and_fail_mask(data):
    targets = build_temporal_targets(data)
    assert np.all(targets.event_week <= targets.observation_week)
    assert set(np.unique(targets.withdrawal_event)).issubset({0.0, 1.0})
    withdrawn = targets.withdrawal_event == 1
    assert np.all(targets.outcome_target[withdrawn] == -1)


def test_ranking_pairs_and_gate_are_registered():
    gate = json.loads(
        (ARTIFACT_ROOT / "prediction/ranking/gate.json").read_text(encoding="utf-8")
    )
    metrics = json.loads(
        (ARTIFACT_ROOT / "prediction/ranking/fold_metrics.json").read_text(encoding="utf-8")
    )
    assert gate["pair_contract"].startswith("same_module_presentation_progress")
    assert all(row["pair_count"] > 0 for row in metrics)


def test_protected_versions_unchanged():
    assert protected_hash_status()["pass"]
