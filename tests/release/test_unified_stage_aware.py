from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from src.models.uci_components import _UCITemporalEncoder
from src.pipelines import uci as unified
from src.pipelines import uci_support as tf

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def mat_views():
    return unified.build_stage_views(tf._load_uci("student_mat"))


def test_stage_view_contract_has_three_aligned_views(mat_views) -> None:
    assert set(mat_views) == set(unified.STAGES)
    reference = mat_views[unified.STAGES[0]]
    for bundle in mat_views.values():
        bundle.validate()
        assert np.array_equal(bundle.record_id, reference.record_id)
        assert np.array_equal(bundle.outer_fold, reference.outer_fold)
        assert np.array_equal(bundle.target, reference.target)


def test_s0_contains_no_grade_signal(mat_views) -> None:
    bundle = mat_views["S0_EARLY_NO_GRADE"]
    assert np.count_nonzero(bundle.temporal) == 0
    assert np.count_nonzero(bundle.availability_mask) == 0


def test_s1_has_one_available_timestep_and_no_g2(mat_views) -> None:
    bundle = mat_views["S1_MID_G1_ONLY"]
    assert np.all(bundle.availability_mask[:, 0] == 1)
    assert np.all(bundle.availability_mask[:, 1] == 0)
    assert np.count_nonzero(bundle.temporal[:, 1, :]) == 0


def test_s2_has_two_available_timesteps(mat_views) -> None:
    bundle = mat_views["S2_LATE_G1_G2"]
    assert np.all(bundle.availability_mask == 1)


def test_g3_is_target_only(mat_views) -> None:
    for bundle in mat_views.values():
        assert "G3" not in bundle.context.columns
        assert "G1" not in bundle.context.columns
        assert "G2" not in bundle.context.columns


def test_g3_target_boundaries_are_frozen() -> None:
    assert tf.encode_uci_target([9, 10, 14, 15, 20]).tolist() == [
        0,
        1,
        1,
        2,
        2,
    ]


def test_frozen_fold_is_identical_across_stages(mat_views) -> None:
    hashes = {
        hashlib.sha256(bundle.outer_fold.tobytes()).hexdigest()
        for bundle in mat_views.values()
    }
    assert len(hashes) == 1


def test_tabular_expansion_is_stage_balanced(mat_views) -> None:
    index = np.arange(17)
    _, target, stages = unified._expand_tabular(mat_views, index)
    assert len(target) == 51
    assert {stage: int((stages == stage).sum()) for stage in unified.STAGES} == {
        stage: 17 for stage in unified.STAGES
    }


def test_one_model_identity_per_dataset_family() -> None:
    assert unified._model_id("student_mat", "cnn_bilstm") == "cnn_bilstm_mat"
    assert unified._model_id("student_por", "cnn_bilstm") == "cnn_bilstm_por"
    assert len(
        {
            unified._model_id(dataset, family)
            for dataset in unified.DATASETS
            for family in unified.MODELS
        }
    ) == 20


def test_training_run_key_has_no_stage_dimension() -> None:
    config = {"alpha": 1}
    run = unified._run_id("student_mat", "mlp", 0, 42, config)
    assert run == unified._run_id("student_mat", "mlp", 0, 42, config)
    assert all(stage not in run for stage in unified.STAGES)


def test_bilstm_s1_is_invariant_to_masked_future_values() -> None:
    torch.manual_seed(7)
    encoder = _UCITemporalEncoder(
        7,
        {
            "temporal_variant": "cnn_bilstm",
            "input_projection": 8,
            "cnn_channels": 8,
            "lstm_hidden": 8,
            "dropout": 0.0,
        },
    ).eval()
    first = torch.randn(4, 2, 7)
    second = first.clone()
    second[:, 1, :] = torch.randn(4, 7) * 100
    mask = torch.tensor([[1.0, 0.0]] * 4)
    with torch.no_grad():
        assert torch.allclose(
            encoder(first, mask), encoder(second, mask), atol=1e-7, rtol=0
        )


def test_bilstm_s0_is_exact_zero() -> None:
    encoder = _UCITemporalEncoder(
        7,
        {
            "temporal_variant": "cnn_bilstm",
            "input_projection": 8,
            "cnn_channels": 8,
            "lstm_hidden": 8,
            "dropout": 0.0,
        },
    ).eval()
    with torch.no_grad():
        output = encoder(torch.randn(3, 2, 7), torch.zeros(3, 2))
    assert torch.count_nonzero(output) == 0


def test_protocol_prohibits_outer_tuning_transfer_and_resampling() -> None:
    protocol = unified._protocol()
    assert protocol["training"]["outer_used_for_tuning"] is False
    assert protocol["training"]["best_seed_selection"] is False
    assert protocol["training"]["transfer_learning"] == "prohibited"
    assert protocol["training"]["pretrained_checkpoint"] == "prohibited"
    assert protocol["training"]["synthetic_resampling"] == "NONE"


def test_grade_band_reference_is_not_model_identity() -> None:
    protocol = unified._protocol()
    assert protocol["reference"]["model_identity"] is False
    assert "grade_band_reference" not in unified.MODELS


def test_oulad_and_future_are_frozen() -> None:
    protocol = unified._protocol()
    assert protocol["oulad"]["stage"] == "F2_MIDDLE"
    assert protocol["oulad"]["retraining"] == "prohibited"
    assert protocol["future_oulad"] == "LOCKED_NOT_EXECUTED"


def test_official_source_checksums_still_match_pre_refactor_guard() -> None:
    guard = json.loads(
        (
            ROOT
            / "artifacts"
            / "final"
            / "protocol_snapshots"
            / "pre_unified_scientific_freeze.json"
        ).read_text(encoding="utf-8")
    )
    frozen_scientific_outputs = {
        "artifacts/final/final_results.json",
        "artifacts/final/final_results.csv",
    }
    for relative, expected in guard["canonical_sha256"].items():
        if relative not in frozen_scientific_outputs:
            continue
        actual = hashlib.sha256((ROOT / relative).read_bytes()).hexdigest()
        assert actual == expected


def test_no_docx_or_pdf_is_part_of_unified_output_contract() -> None:
    assert all(
        suffix not in {".docx", ".pdf"}
        for suffix in (path.suffix.lower() for path in unified.OUT.rglob("*"))
    )


def test_generated_prediction_authority_has_expected_rows() -> None:
    predictions = unified.pd.read_parquet(unified.OUT / "predictions.parquet")
    seeds = unified.pd.read_parquet(unified.OUT / "seed_predictions.parquet")
    assert len(predictions) == 31_320
    assert len(seeds) == 156_600
    assert not predictions.duplicated(
        ["dataset", "model_id", "prediction_stage", "record_id"]
    ).any()


def test_generated_metric_authority_has_expected_rows() -> None:
    stage = unified.pd.read_csv(unified.OUT / "stage_metrics.csv")
    overall = unified.pd.read_csv(unified.OUT / "overall_metrics.csv")
    assert len(stage) == 60
    assert len(overall) == 20
    assert stage.groupby(["dataset", "prediction_stage"]).size().eq(10).all()


def test_each_training_run_maps_three_stages_to_one_checkpoint() -> None:
    manifest = json.loads(
        (unified.OUT / "checkpoint_stage_mapping.json").read_text(
            encoding="utf-8"
        )
    )
    assert manifest["mapping_count"] == 1500
    rows = unified.pd.DataFrame(manifest["rows"])
    grouped = rows.groupby("training_run_id")
    assert grouped.size().eq(3).all()
    assert grouped["checkpoint"].nunique().eq(1).all()
    assert grouped["checkpoint_sha256"].nunique().eq(1).all()


def test_training_manifest_contains_500_stage_independent_runs() -> None:
    manifest = json.loads(
        (unified.OUT / "training_run_manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["training_run_count"] == 500
    assert all(row["one_training_run_all_stages"] for row in manifest["rows"])
    assert all("stage" not in row["config_hash"].lower() for row in manifest["rows"])


def test_inner_trials_never_use_outer_rows() -> None:
    trials = unified.pd.read_csv(unified.OUT / "inner_trials.csv")
    selected = unified.pd.read_csv(unified.OUT / "selected_configs.csv")
    assert not trials["outer_rows_used_for_selection"].any()
    assert not selected["outer_rows_used_for_selection"].any()
    assert len(selected) == 100


def test_joint_overall_bootstrap_is_base_record_paired() -> None:
    frame = unified.pd.read_csv(unified.OUT / "bootstrap_overall.csv")
    assert len(frame) == 18
    assert set(frame["replicates"]) == {5000}
    assert set(frame["resampling_unit"]) == {
        "base_record_with_all_three_stage_views"
    }


def test_final_authorities_include_30_overall_identities() -> None:
    stage = unified.pd.read_csv(
        ROOT / "artifacts" / "final" / "final_stage_results.csv"
    )
    overall = unified.pd.read_csv(
        ROOT / "artifacts" / "final" / "final_overall_results.csv"
    )
    assert len(overall) == 30
    assert len(stage.loc[stage["dataset"].isin(unified.DATASETS)]) == 60
    assert len(stage.loc[stage["dataset"] == "oulad"]) == 40
    assert set(stage.loc[stage["dataset"] == "oulad", "prediction_stage"]) == {
        "E1_EARLY_20PCT",
        "E2_EARLY_35PCT",
        "M1_MIDDLE_FROZEN",
        "L1_LATE_75PCT",
    }


def test_replacement_database_evidence_is_ready_without_cutover() -> None:
    payload = json.loads(
        (
            ROOT
            / "artifacts"
            / "final"
            / "database"
            / "unified_database_replacement_validation.json"
        ).read_text(encoding="utf-8")
    )
    assert payload["status"] == "READY_FOR_DATABASE_CUTOVER"
    assert payload["canonical_database_modified"] is False
    assert payload["replacement_validation"]["status"] == "PASS"
    assert payload["replacement_validation"]["counts"]["models"] == 30
    assert payload["replacement_validation"]["counts"]["risk_profiles"] == 15378
    assert payload["replacement_validation"]["counts"]["plans"] == 15378
    assert payload["replacement_validation"]["counts"]["actions"] == 27355


def test_public_pipeline_paths_are_non_versioned() -> None:
    assert (ROOT / "src" / "pipelines" / "uci.py").is_file()
    assert (ROOT / "src" / "pipelines" / "oulad.py").is_file()
    assert (ROOT / "configs" / "final" / "uci_prediction.yaml").is_file()
    assert (ROOT / "configs" / "final" / "oulad_prediction.yaml").is_file()
    assert not (ROOT / "src" / "studies" / "teacher_feedback.py").exists()
    assert not (ROOT / "artifacts" / "history").is_dir()
    assert not (ROOT / "artifacts" / "refactor").is_dir()
