from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
from sklearn.metrics import f1_score

from src.studies.oulad_v2.data import build_inner_manifest, load_v2_data, manifest_indices
from src.studies.oulad_v2.metrics import choose_thresholds, grouped_bootstrap_prediction_delta, prediction_frame_metrics
from src.studies.oulad_v2.models import OULADV2Net, prepare_inputs, set_deterministic_seed


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = json.loads((ROOT / "configs/oulad_deep_v2_protocol.yaml").read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture(scope="module")
def data():
    return load_v2_data(ROOT / "data/processed/study_c_oulad", PROTOCOL)


def test_v1_artifacts_are_immutable():
    v1 = PROTOCOL["v1_immutable"]
    assert _sha256(ROOT / v1["protocol_path"]) == v1["protocol_sha256"]
    assert _sha256(ROOT / v1["artifact_path"] / "metrics_by_model_forecast.csv") == v1["metrics_sha256"]
    assert _sha256(ROOT / v1["artifact_path"] / "oof_predictions.parquet") == v1["oof_predictions_sha256"]


def test_f2_only_pilot_and_future_inaccessible():
    assert PROTOCOL["data"]["forecast_id"] == "F2_MIDDLE"
    assert PROTOCOL["future_policy"]["available_during_selection"] is False
    assert PROTOCOL["candidate_registry"]["mandatory"]["V2-MLF"]["trainable"] is False


def test_new_seeds_are_fixed():
    assert PROTOCOL["seeds"] == [42, 2026, 3407]


def test_global_student_grouping_and_outer_disjointness(data):
    for outer_fold in range(3):
        train, validation = data.outer_indices(outer_fold)
        assert not (set(data.groups[train]) & set(data.groups[validation]))


def test_inner_outer_disjointness(data):
    for outer_fold in range(3):
        manifest = build_inner_manifest(data, outer_fold, 3407, 2)
        _, outer_validation = data.outer_indices(outer_fold)
        outer_validation_students = set(data.groups[outer_validation])
        for inner_fold in range(2):
            train, validation = manifest_indices(data, manifest, inner_fold)
            assert not (set(data.groups[train]) & set(data.groups[validation]))
            assert not (outer_validation_students & set(data.groups[train]))
            assert not (outer_validation_students & set(data.groups[validation]))


def test_aggregate_schema_is_exact_and_not_flattened(data):
    assert len(data.aggregate_columns) == 161
    assert not any(column.startswith("week_") for column in data.aggregate_columns)
    assert "final_result" not in data.aggregate_columns
    assert "target_at_risk" not in data.aggregate_columns


def test_a0_h3c_aggregate_input_parity(data):
    train, validation = data.outer_indices(0)
    train, validation = train[:100], validation[:20]
    a0 = prepare_inputs(data, train, validation, "V2-A0")
    h3c = prepare_inputs(data, train, validation, "V2-H3C")
    np.testing.assert_allclose(a0.aggregate, h3c.aggregate)
    np.testing.assert_allclose(a0.static, h3c.static)


def test_h2t_h3c_temporal_input_parity(data):
    train, validation = data.outer_indices(0)
    train, validation = train[:100], validation[:20]
    h2t = prepare_inputs(data, train, validation, "V2-H2T")
    h3c = prepare_inputs(data, train, validation, "V2-H3C")
    np.testing.assert_allclose(h2t.sequence, h3c.sequence)
    np.testing.assert_array_equal(h2t.mask, h3c.mask)


def test_preprocessing_is_fit_on_explicit_training_rows(data):
    train, validation = data.outer_indices(0)
    prepared = prepare_inputs(data, train[:100], validation[:20], "V2-H3C")
    assert prepared.preprocessors.sequence_mean is not None
    assert prepared.preprocessors.aggregate is not None
    assert prepared.preprocessors.static is not None


def test_threshold_contract_is_probability_based_and_deterministic():
    y = np.asarray([0, 0, 0, 1, 1, 1])
    probability = np.asarray([0.1, 0.2, 0.7, 0.4, 0.8, 0.9])
    first = choose_thresholds(y, probability)
    second = choose_thresholds(y, probability)
    assert first == second
    assert 0.05 <= first["macro_threshold"] <= 0.95
    assert first["operational_feasible"]


def test_precision_constraint_is_frozen():
    assert PROTOCOL["metrics"]["precision_constraint"] == 0.75


def test_h3c_is_concat_only_no_gating_or_attention():
    source = (ROOT / "src/studies/oulad_v2/models.py").read_text(encoding="utf-8")
    h3c_block = source[source.index('elif candidate_id == "V2-H3C"'):]
    assert "nn.Linear" in h3c_block
    assert "attention" not in h3c_block.lower()
    assert "gating" not in h3c_block.lower()
    assert "nn.Sigmoid" not in h3c_block


def test_model_parameter_guardrail(data):
    train, validation = data.outer_indices(0)
    prepared = prepare_inputs(data, train[:100], validation[:20], "V2-H3C")
    temporal = {"conv_channels": 48, "kernel_size": 5, "lstm_hidden": 64, "lstm_layers": 2, "dropout": 0.35}
    aggregate = {"aggregate_hidden_1": 128, "aggregate_hidden_2": 64, "dropout": 0.35}
    model = OULADV2Net("V2-H3C", 16, prepared.aggregate.shape[1], prepared.static.shape[1], temporal, aggregate)
    assert sum(parameter.numel() for parameter in model.parameters()) < 300_000


def test_padding_mask_and_temporal_output_shape(data):
    train, validation = data.outer_indices(0)
    prepared = prepare_inputs(data, train[:100], validation[:8], "V2-H2T")
    temporal = {"conv_channels": 24, "kernel_size": 3, "lstm_hidden": 32, "lstm_layers": 1, "dropout": 0.2, "static_hidden": 32, "fusion_hidden": 32}
    model = OULADV2Net("V2-H2T", 16, 1, prepared.static.shape[1], temporal, None)
    with torch.no_grad():
        output = model(
            torch.from_numpy(prepared.sequence), torch.from_numpy(prepared.lengths), torch.from_numpy(prepared.mask),
            torch.from_numpy(prepared.aggregate), torch.from_numpy(prepared.static),
        )
    assert output.shape == (8,)
    assert torch.isfinite(output).all()


def test_seed_initialization_is_deterministic():
    set_deterministic_seed(42); first = torch.rand(5)
    set_deterministic_seed(42); second = torch.rand(5)
    torch.testing.assert_close(first, second)


def test_pooled_oof_metric_recomputation():
    frame = pd.DataFrame(
        {
            "target_at_risk": [0, 0, 1, 1], "probability": [0.1, 0.7, 0.6, 0.9],
            "predicted_label": [0, 1, 1, 1], "operational_prediction": [0, 1, 1, 1],
            "operational_feasible": [True] * 4,
        }
    )
    metrics = prediction_frame_metrics(frame)
    assert metrics["macro_f1"] == pytest.approx(f1_score(frame.target_at_risk, frame.predicted_label, average="macro"))


def test_grouped_bootstrap_is_paired_by_student():
    left = pd.DataFrame({"record_id": ["a", "b", "c", "d"], "id_student": [1, 1, 2, 3], "target_at_risk": [0, 1, 0, 1], "predicted_label": [0, 1, 0, 1]})
    right = pd.DataFrame({"record_id": ["a", "b", "c", "d"], "id_student": [1, 1, 2, 3], "target_at_risk": [0, 1, 0, 1], "predicted_label": [1, 1, 0, 0]})
    result = grouped_bootstrap_prediction_delta(left, right, resamples=100, seed=42)
    assert result["mean_delta"] > 0
    assert result["resamples"] == 100


def test_runner_has_no_future_prediction_input():
    source = (ROOT / "scripts/run_oulad_deep_v2.py").read_text(encoding="utf-8")
    assert 'future_predictions.parquet' not in source
    assert 'available_during_selection"]:' in source


def test_adaptive_envelope_forbids_scientific_changes():
    forbidden = set(PROTOCOL["search"]["adaptive_envelope"]["forbidden"])
    assert {"target", "cohort", "split", "primary_metric", "future_policy", "seeds", "gate"} <= forbidden


def test_partial_run_cannot_validate_complete():
    source = (ROOT / "scripts/run_oulad_deep_v2.py").read_text(encoding="utf-8")
    assert 'len(metadata) != 36' in source
