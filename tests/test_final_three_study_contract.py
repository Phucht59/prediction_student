from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from scripts.validate_release import (
    STUDY_A,
    STUDY_B,
    STUDY_C,
    validate_documents_and_source_scope,
    validate_metrics,
    verify_checksum_manifest,
)
from src.estimator_factory import resolve_student_grade_neural_config
from src.models import create_student_grade_model


ROOT = Path(__file__).resolve().parents[1]


def test_readme_and_project_cover_all_three_datasets_and_final_verdicts():
    for filename in ("README.md", "PROJECT.md"):
        text = (ROOT / filename).read_text(encoding="utf-8")
        for required in ("student-mat", "student-por", "OULAD"):
            assert required in text
        assert "0.8988" in text
        assert "0.8698" in text
        assert "0.8311" in text
        assert "PRACTICAL_TIE" in text
        assert "expert_validation = PENDING" in text or "Expert validation: **PENDING**" in text


def test_final_release_metrics_recompute_from_all_three_official_tables():
    assert validate_metrics() == {"study_a_models": 5, "study_b_models": 10, "study_c_models": 8}


def test_official_three_study_checksum_manifests_remain_valid():
    assert verify_checksum_manifest(STUDY_A, STUDY_A / "artifact_checksums.json") > 0
    assert verify_checksum_manifest(STUDY_B, STUDY_B / "artifact_checksums.json") > 0
    assert verify_checksum_manifest(STUDY_C, STUDY_C / "artifact_checksums.json") > 0


def test_obsolete_strategy_entrypoints_are_absent_from_final_source():
    validate_documents_and_source_scope()
    assert not (ROOT / "MODEL_IMPROVEMENT_PLAN_V3.md").exists()
    assert not (ROOT / "SCIENTIFIC_PROTOCOL_V2.md").exists()


def test_project_cli_is_the_single_routine_entrypoint_and_status_is_read_only():
    result = subprocess.run(
        [sys.executable, str(ROOT / "project.py"), "status"],
        cwd=ROOT,
        check=False,
        text=True,
        capture_output=True,
    )
    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert payload["all_official_bundles_available"] is True
    assert set(payload) >= {"student_mat", "student_por", "oulad"}

    source = (ROOT / "project.py").read_text(encoding="utf-8")
    assert "fit_candidate" not in source
    assert "Optuna" not in source
    assert len(list((ROOT / "scripts").glob("*.py"))) <= 11


def test_student_grade_model_has_generic_active_module_and_same_contract():
    config = resolve_student_grade_neural_config(
        "N0",
        {
            "learning_rate": 0.001,
            "weight_decay": 0.00001,
            "batch_size": 32,
            "oversample_method": "none",
            "class_weight_mode": "none",
            "loss": "cross_entropy",
            "smote_ratio": 1.0,
            "resampling_k_neighbors": 5,
            "cnn_channels": 8,
            "cnn_kernel_size": 1,
            "lstm_hidden_dim": 8,
            "dropout": 0.1,
            "sequence_dropout": 0.0,
            "max_epochs": 24,
            "patience": 4,
            "normalization": "none",
            "hidden_dim": 8,
            "num_layers": 1,
        },
        suggested_parameters={},
    )
    model = create_student_grade_model(config)
    assert model.__class__.__module__ == "src.models.student_grade"
    assert config["feature_contract"]["sequence_columns"] == ["G1", "G2"]
