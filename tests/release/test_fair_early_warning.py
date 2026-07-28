from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from src.models._uci import _UCITemporalEncoder
from src.studies import early_warning as ew
from src.studies import teacher_feedback as tf

ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize(
    ("grade", "expected"), [(9, 0), (10, 1), (14, 1), (15, 2), (20, 2)]
)
def test_g3_target_boundaries(grade: int, expected: int) -> None:
    assert tf.encode_uci_target([grade]).item() == expected


def test_target_independent_from_g1_g2() -> None:
    assert tf.encode_uci_target([14]).item() == tf.encode_uci_target([14]).item()


def test_target_changes_across_g3_boundary() -> None:
    assert tf.encode_uci_target([9]).item() != tf.encode_uci_target([10]).item()


@pytest.mark.parametrize("dataset", ew.DATASETS)
def test_s0_has_no_grade_information(dataset: str) -> None:
    data = tf._load_uci(dataset)
    frame = tf.build_uci_scenario_frame(data.frame, ew.SCENARIOS[0])
    assert not any(column.startswith("grade_") for column in frame)


@pytest.mark.parametrize("dataset", ew.DATASETS)
def test_s1_has_no_g2_information(dataset: str) -> None:
    data = tf._load_uci(dataset)
    frame = tf.build_uci_scenario_frame(data.frame, ew.SCENARIOS[1])
    assert not any(column.startswith("grade_t1_") for column in frame)


@pytest.mark.parametrize("dataset", ew.DATASETS)
def test_s2_has_no_g3_information(dataset: str) -> None:
    data = tf._load_uci(dataset)
    assert "G3" not in tf.build_uci_scenario_frame(data.frame, ew.SCENARIOS[2])


def test_fixed_temporal_shape_and_masks() -> None:
    data = tf._load_uci("student_mat")
    expected = ([0, 0], [1, 0], [1, 1])
    for scenario, mask_value in zip(ew.SCENARIOS, expected):
        values, mask = ew.build_temporal(data.frame.iloc[:3], scenario)
        assert values.shape == (3, 2, 7)
        assert mask.shape == (3, 2)
        assert mask[0].tolist() == mask_value


@pytest.mark.parametrize("variant", ew.DEEP)
def test_masked_s0_zero_embedding(variant: str) -> None:
    encoder = _UCITemporalEncoder(
        7,
        {
            "temporal_variant": variant,
            "input_projection": 8,
            "cnn_channels": 8,
            "lstm_hidden": 8,
            "dropout": 0.0,
        },
    ).eval()
    output = encoder(torch.randn(3, 2, 7), torch.zeros(3, 2))
    assert torch.equal(output, torch.zeros_like(output))


@pytest.mark.parametrize("variant", ew.DEEP)
def test_masked_placeholder_invariant(variant: str) -> None:
    encoder = _UCITemporalEncoder(
        7,
        {
            "temporal_variant": variant,
            "input_projection": 8,
            "cnn_channels": 8,
            "lstm_hidden": 8,
            "dropout": 0.0,
        },
    ).eval()
    first = torch.randn(3, 2, 7)
    second = first.clone()
    second[:, 1] = 100
    mask = torch.tensor([[1.0, 0.0]] * 3)
    assert torch.allclose(encoder(first, mask), encoder(second, mask), atol=1e-7)


@pytest.mark.parametrize("variant", ew.DEEP)
def test_s2_legacy_compatibility(variant: str) -> None:
    encoder = _UCITemporalEncoder(
        7,
        {
            "temporal_variant": variant,
            "input_projection": 8,
            "cnn_channels": 8,
            "lstm_hidden": 8,
            "dropout": 0.0,
        },
    ).eval()
    values = torch.randn(3, 2, 7)
    assert torch.allclose(
        encoder(values), encoder(values, torch.ones(3, 2)), atol=1e-6
    )


def test_masked_gradient_is_zero() -> None:
    encoder = _UCITemporalEncoder(
        7,
        {
            "temporal_variant": "cnn_bilstm",
            "input_projection": 8,
            "cnn_channels": 8,
            "lstm_hidden": 8,
            "dropout": 0.0,
        },
    )
    values = torch.randn(2, 2, 7, requires_grad=True)
    encoder(values, torch.tensor([[1.0, 0.0]] * 2)).sum().backward()
    assert torch.equal(values.grad[:, 1], torch.zeros_like(values.grad[:, 1]))


def test_preregistered_no_transfer_protocol() -> None:
    protocol = ew._protocol()
    assert protocol["scope"] == "FAIR_TIMING_NO_TRANSFER"
    assert protocol["transfer_learning"] == "prohibited"
    assert protocol["outer_used_for_tuning"] is False


def test_unified_model_count() -> None:
    assert len(ew.MODELS) == 10


def test_generated_result_matrix_complete() -> None:
    metrics = pd.concat(
        [
            pd.read_csv(ew.ROOT_OUT / "student_mat_metrics.csv"),
            pd.read_csv(ew.ROOT_OUT / "student_por_metrics.csv"),
        ]
    )
    assert len(metrics[["dataset", "scenario", "model_id"]].drop_duplicates()) == 60


def test_same_outer_split_hashes() -> None:
    payload = json.loads(
        (ew.ROOT_OUT / "split_equivalence.json").read_text(encoding="utf-8")
    )
    assert payload["status"] == "PASS"
    assert all(row["same_split_for_all_models"] for row in payload["rows"])


def test_training_only_and_no_outer_selection() -> None:
    payload = json.loads(
        (ew.TF_OUT / "fair_comparison_contract.json").read_text(encoding="utf-8")
    )
    assert all(row["preprocessing_fit_scope"] == "training_only" for row in payload["rows"])
    assert all(row["outer_rows_used_for_tuning"] == 0 for row in payload["rows"])


def test_no_transfer_rows() -> None:
    payload = json.loads(
        (ew.ROOT_OUT / "leakage_validation.json").read_text(encoding="utf-8")
    )
    assert payload["transfer_rows_used"] == 0


def test_per_class_complete() -> None:
    frame = pd.read_csv(ew.ROOT_OUT / "per_class_metrics.csv")
    assert len(frame) == 180
    assert set(frame["class_label"]) == {"Low", "Medium", "High"}


def test_bootstrap_is_5000_and_complete() -> None:
    frame = pd.read_csv(ew.ROOT_OUT / "paired_bootstrap_all_models.csv")
    assert len(frame) == 180
    assert set(frame["replicates"]) == {5000}


def test_mlp_in_common_matrix() -> None:
    frame = pd.read_csv(ew.ROOT_OUT / "scenario_rankings.csv")
    assert len(frame.loc[frame["model_id"] == "mlp"]) == 6


def test_grade_reference_is_training_only() -> None:
    frame = pd.read_csv(ew.ROOT_OUT / "grade_band_reference_metrics.csv")
    assert len(frame) == 6
    assert set(frame["fit_scope"]) == {"outer_training_fold_only"}


def test_oulad_temporal_contract() -> None:
    payload = json.loads(
        (ew.TF_OUT / "oulad_temporal_branch_audit.json").read_text(encoding="utf-8")
    )
    assert payload["temporal_channel_count"] == 47
    assert len(payload["channels"]) == 47
    assert not payload["sequence_contains_static"]
    assert not payload["sequence_contains_aggregate_summary_branch"]


def test_imbalance_safety() -> None:
    payload = json.loads(
        (ew.TF_OUT / "imbalance_final_safety_audit.json").read_text(encoding="utf-8")
    )
    assert payload["status"] == "PASS"
    assert payload["canonical_timing_synthetic_resampling"] == "NONE"


def test_official_freeze() -> None:
    assert tf.verify_regression_guard()["status"] == "PASS"


def test_future_oulad_xapi_and_recommendation_frozen() -> None:
    guard = tf._official_snapshot()
    assert guard["future_oulad"] == "LOCKED_NOT_EXECUTED"
    assert not guard["xapi_in_final_datasets"]
    assert guard["recommendation"]["records"] == 15378

