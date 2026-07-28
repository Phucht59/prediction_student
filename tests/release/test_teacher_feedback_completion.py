from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.final_release.build import ROOT
from src.final_release.catalog import COMPARISON_MODELS
from src.studies.teacher_feedback import (
    build_uci_scenario_frame,
    encode_uci_target,
)


TF_ROOT = ROOT / "artifacts" / "final" / "teacher_feedback_validation"
TIMING_ROOT = ROOT / "artifacts" / "final" / "uci_timing_scenarios"


def _sample_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "failures": [0, 1],
            "studytime": [2, 3],
            "schoolsup": ["yes", "no"],
            "famsup": ["no", "yes"],
            "paid": ["no", "yes"],
            "activities": ["yes", "no"],
            "internet": ["yes", "yes"],
            "higher": ["yes", "yes"],
            "traveltime": [1, 2],
            "freetime": [3, 4],
            "goout": [2, 3],
            "health": [5, 4],
            "G1": [9, 15],
            "G2": [10, 14],
            "G3": [11, 16],
        }
    )


def test_target_threshold_boundaries() -> None:
    assert encode_uci_target([9, 10, 14, 15, 20]).tolist() == [0, 1, 1, 2, 2]


def test_g3_is_excluded_from_every_scenario() -> None:
    frame = _sample_frame()
    for scenario in (
        "S0_EARLY_NO_GRADE",
        "S1_MID_G1_ONLY",
        "S2_LATE_G1_G2",
    ):
        assert "G3" not in build_uci_scenario_frame(frame, scenario).columns


def test_s0_excludes_raw_and_derived_grades() -> None:
    columns = build_uci_scenario_frame(
        _sample_frame(), "S0_EARLY_NO_GRADE"
    ).columns
    assert "G1" not in columns and "G2" not in columns
    assert not any(column.startswith("grade_") for column in columns)


def test_s1_excludes_g2_and_g2_derived_features() -> None:
    columns = build_uci_scenario_frame(
        _sample_frame(), "S1_MID_G1_ONLY"
    ).columns
    assert "G2" not in columns
    assert not any(column.startswith("grade_t1_") for column in columns)
    assert "grade_t0_normalized_grade" in columns


def test_s2_reproduces_frozen_two_timestep_information_contract() -> None:
    frame = _sample_frame()
    result = build_uci_scenario_frame(frame, "S2_LATE_G1_G2")
    grade_columns = [column for column in result if column.startswith("grade_")]
    assert len(grade_columns) == 14
    assert result.loc[0, "grade_t0_normalized_grade"] == 9 / 20
    assert result.loc[0, "grade_t1_normalized_grade"] == 10 / 20
    assert result.loc[0, "grade_t1_signed_change_from_G1"] == 1 / 20
    assert result.loc[1, "grade_t1_change_direction"] == -1


def test_split_hashes_are_identical_across_models() -> None:
    payload = json.loads(
        (TF_ROOT / "split_equivalence.json").read_text(encoding="utf-8")
    )
    assert payload["status"] == "PASS"
    assert payload["all_models_same_split_within_dataset_scenario"]
    assert payload["outer_rows_in_inner_training"] == 0


def test_no_validation_rows_or_sampler_fit() -> None:
    payload = json.loads(
        (TF_ROOT / "evaluation_contract.json").read_text(encoding="utf-8")
    )
    assert payload["outer_validation_used_for_tuning"] is False
    assert payload["preprocessing_fit"] == "outer_train_or_inner_train_only"
    audit = json.loads(
        (TF_ROOT / "imbalance_safety_audit.json").read_text(encoding="utf-8")
    )
    assert audit["current_executable_sampler_matches"] == []


def test_no_synthetic_resampling_on_oulad_tensor() -> None:
    audit = json.loads(
        (TF_ROOT / "imbalance_safety_audit.json").read_text(encoding="utf-8")
    )
    check = audit["questions"][
        "synthetic_oversampling_on_raw_oulad_temporal_tensor"
    ]
    assert check == {"answer": False, "status": "PASS"}
    assert audit["new_study"]["oulad_tensor_supplied_to_mlp"] is False


def test_mlp_present_and_ten_models_per_dataset() -> None:
    assert ("mlp", "MLP") in COMPARISON_MODELS
    payload = json.loads(
        (ROOT / "artifacts" / "final" / "final_results.json").read_text(
            encoding="utf-8"
        )
    )
    assert all(len(dataset["models"]) == 10 for dataset in payload["datasets"].values())
    assert all(
        any(model["model_id"] == "mlp" for model in dataset["models"])
        for dataset in payload["datasets"].values()
    )


def test_official_metrics_and_recommendation_are_unchanged() -> None:
    payload = json.loads(
        (ROOT / "artifacts" / "final" / "final_results.json").read_text(
            encoding="utf-8"
        )
    )
    expected = {
        "student_mat": 0.9014601961315334,
        "student_por": 0.8622587167738002,
        "oulad": 0.8280835945631038,
    }
    for dataset, value in expected.items():
        official = next(
            row
            for row in payload["datasets"][dataset]["models"]
            if row["model_id"] == "cnn_bilstm"
        )
        assert official["metrics"]["macro_f1"]["value"] == value
    recommendation = payload["recommendation"]["metrics"]
    assert recommendation["records"]["value"] == 15378
    assert recommendation["generated"]["value"] == 10953
    assert recommendation["partial_evidence"]["value"] == 1209
    assert recommendation["abstained"]["value"] == 3216


def test_future_oulad_locked_expert_pending_and_xapi_absent() -> None:
    payload = json.loads(
        (ROOT / "artifacts" / "final" / "final_results.json").read_text(
            encoding="utf-8"
        )
    )
    assert payload["future_oulad"] == "LOCKED_NOT_EXECUTED"
    assert (
        payload["recommendation"]["expert_status"]["value"]
        == "PENDING_EXPERT_LABELS"
    )
    assert "xapi" not in payload["datasets"]


def test_deep_timing_is_not_misrepresented() -> None:
    payload = json.loads(
        (TF_ROOT / "deep_timing_feasibility.json").read_text(encoding="utf-8")
    )
    assert payload["status"] == "NOT_RUN_ARCHITECTURE_NOT_COMPARABLE"
    assert payload["decision"] == "CASE_B"
    assert payload["findings"]["official_temporal_length"] == 2
    assert payload["findings"]["explicit_timestep_mask_supported"] is False
    assert payload["prohibited_actions_confirmed"]["fake_zero_filled_G2_used"] is False
    assert (
        payload["prohibited_actions_confirmed"][
            "context_only_model_called_cnn_bilstm"
        ]
        is False
    )


def test_safe_baseline_revalidation_is_fully_disclosed() -> None:
    payload = json.loads(
        (TF_ROOT / "baseline_revalidation.json").read_text(encoding="utf-8")
    )
    assert payload["status"] == "PASS"
    assert len(payload["rows"]) == 12
    assert payload["protocol"]["same_frozen_outer_splits"] is True
    assert payload["protocol"]["preprocessing_fit_training_only"] is True
    assert payload["protocol"]["synthetic_resampling_used"] is False
    assert payload["protocol"]["official_cnn_bilstm_affected"] is False


def test_pre_closure_snapshot_freezes_canonical_state() -> None:
    payload = json.loads(
        (TF_ROOT / "pre_closure_checksum_snapshot.json").read_text(
            encoding="utf-8"
        )
    )
    assert payload["closure_base_commit"] == (
        "248ca8ca7f5f22a6e470dbc5dd1dc11c52231a31"
    )
    assert payload["comparator_contract"]["model_dataset_identities"] == 30
    assert payload["recommendation"]["risk_profiles"] == 15378
    assert payload["recommendation"]["plan_objects"] == 15378
    assert payload["recommendation"]["actions"] == 27355
    assert payload["future_oulad"] == "LOCKED_NOT_EXECUTED"


def test_artifact_checksums_are_deterministic_and_valid() -> None:
    import hashlib

    for manifest_path in (
        TIMING_ROOT / "checksums.json",
        TF_ROOT / "checksum_manifest.json",
    ):
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        for relative, expected in manifest["files"].items():
            digest = hashlib.sha256((ROOT / relative).read_bytes()).hexdigest()
            assert digest == expected
