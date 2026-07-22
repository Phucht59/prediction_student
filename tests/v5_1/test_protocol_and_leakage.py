from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from src.studies.v5_1.common.artifacts import safe_v5_1_root
from src.studies.v5_1.common.protocol import ROOT, load_protocol, sha256_file
from src.studies.v5_1.common.uci_data import PRIMARY_CONTEXT_FEATURES, build_temporal_features
from src.studies.v5_1.oulad.pretraining import deterministic_week_mask


@pytest.mark.parametrize("name", ["project", "student-mat", "student-por", "oulad"])
def test_every_v5_1_protocol_is_frozen(name: str) -> None:
    filename = "project_v5_1_protocol.yaml" if name == "project" else f"{name.replace('-', '_')}_v5_1.yaml"
    protocol = yaml.safe_load((ROOT / "configs" / "v5_1" / filename).read_text(encoding="utf-8"))
    assert protocol["protocol_status"] == "frozen_before_outer_evaluation"


@pytest.mark.parametrize("dataset", ["student-mat", "student-por"])
def test_uci_source_and_outer_manifest_hashes_are_locked(dataset: str) -> None:
    protocol = load_protocol(dataset)
    assert sha256_file(ROOT / protocol["source"]["path"]) == protocol["source"]["sha256"]
    assert (
        sha256_file(ROOT / protocol["splits"]["outer_manifest"])
        == protocol["splits"]["outer_manifest_sha256"]
    )


def test_v5_checksum_evidence_matches_protocol_snapshot() -> None:
    project = yaml.safe_load(
        (ROOT / "configs" / "v5_1" / "project_v5_1_protocol.yaml").read_text(encoding="utf-8")
    )
    assert (
        sha256_file(ROOT / "artifacts/v5/final/artifact_checksums.json")
        == project["immutability"]["v5_final_checksum_manifest_sha256"]
    )
    assert (
        sha256_file(ROOT / "reports/v5/final/validation_report.json")
        == project["immutability"]["v5_validation_report_sha256"]
    )


def test_future_oulad_is_locked_in_project_and_study_protocols() -> None:
    project = yaml.safe_load(
        (ROOT / "configs" / "v5_1" / "project_v5_1_protocol.yaml").read_text(encoding="utf-8")
    )
    oulad = load_protocol("oulad")
    assert project["future_lock"]["oulad_future_benchmark"] == "LOCKED_NOT_EXECUTED"
    assert oulad["data_contract"]["future_benchmark"] == "LOCKED_NOT_EXECUTED"
    assert set(oulad["data_contract"]["forbidden_roles"]) == {
        "future_candidate",
        "excluded_future_student_overlap",
    }


def test_oulad_stage_gated_compute_amendment_preserves_final_fairness() -> None:
    oulad = load_protocol("oulad")
    search = oulad["search"]
    evaluation = oulad["evaluation"]
    assert search["mode"] == "stage_gated_compute_reduction"
    assert search["component_screening_outer_fold"] == 0
    assert search["round_a_architecture_evaluated_trials_initial"] == 8
    assert search["round_a_architecture_evaluated_trials_max"] == 12
    assert search["round_a_budget_unit"] == "unique_fully_evaluated_configs"
    assert search["round_a_pruning"] == "disabled_full_fold_mean"
    assert search["round_b_outer_fold"] == 0
    assert search["round_b_trials_initial_total"] == 16
    assert search["round_b_trials_max_total"] == 24
    assert evaluation["selected_configuration_reused_across_all_outer_folds"] is True
    assert evaluation["seeds"] == [42, 1201, 2026, 3407, 7319]
    assert evaluation["architecture_ablations"] == ["cnn_only", "bilstm_only"]


def test_v5_1_output_guard_rejects_v4_v5_and_unscoped_paths(tmp_path: Path) -> None:
    valid = safe_v5_1_root(tmp_path / "v5_1" / "study")
    assert valid.is_dir()
    with pytest.raises(RuntimeError):
        safe_v5_1_root(tmp_path / "v5" / "study")
    with pytest.raises(RuntimeError):
        safe_v5_1_root(tmp_path / "unscoped")


def test_temporal_features_are_invariant_to_every_non_g1_g2_column() -> None:
    frame = pd.DataFrame(
        {
            "G1": [5, 10, 15],
            "G2": [7, 9, 18],
            "G3": [1, 10, 20],
            "absences": [0, 5, 100],
            "future_value": [1000, 2000, 3000],
        }
    )
    changed = frame.assign(G3=[20, 20, 20], absences=[99, 99, 99], future_value=[-1, -1, -1])
    np.testing.assert_array_equal(build_temporal_features(frame), build_temporal_features(changed))
    assert "absences" not in PRIMARY_CONTEXT_FEATURES


def test_masked_week_pretraining_never_uses_padding_or_all_valid_weeks() -> None:
    valid = np.array([[1, 1, 1, 1, 0], [1, 1, 0, 0, 0]], dtype=bool)
    selected = deterministic_week_mask(valid, 0.9, seed=42)
    assert not (selected & ~valid).any()
    assert np.all(selected.sum(axis=1) < valid.sum(axis=1))


def test_fixed_seed_registry_matches_all_studies() -> None:
    project = yaml.safe_load(
        (ROOT / "configs" / "v5_1" / "project_v5_1_protocol.yaml").read_text(encoding="utf-8")
    )
    expected = [42, 1201, 2026, 3407, 7319]
    assert project["fixed_seeds"] == expected
    for dataset in ["student-mat", "student-por", "oulad"]:
        assert load_protocol(dataset)["evaluation"]["seeds"] == expected


def test_frozen_v5_validation_remains_pass() -> None:
    report = json.loads((ROOT / "reports/v5/final/validation_report.json").read_text(encoding="utf-8"))
    assert report["status"] == "PASS"
    assert report["future_benchmark"] == "NOT_EXECUTED"
