from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
ARTIFACT_ROOT = ROOT / "artifacts" / "audit" / "phase8"
REPORT_ROOT = ROOT / "reports" / "audit" / "phase8"


def _json(name: str) -> dict:
    return json.loads((ARTIFACT_ROOT / name).read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def test_phase8_required_outputs_exist() -> None:
    artifacts = {
        "h0_endpoint_profile.json",
        "h1_endpoint_profile.json",
        "h0_vs_h1_endpoint_diff.csv",
        "feature_schema_diff.csv",
        "target_population_diff.json",
        "preprocessing_diff.json",
        "training_policy_diff.json",
        "threshold_analysis.json",
        "metric_reproduction.json",
        "error_overlap.csv",
        "root_cause_ranking.csv",
        "recovery_decision.json",
        "phase8_gate.json",
    }
    reports = {
        "PHASE8_SUMMARY.md",
        "PHASE8_H0_VS_H1_FORENSIC.md",
        "PHASE8_FEATURE_SCHEMA.md",
        "PHASE8_TARGET_AND_POPULATION.md",
        "PHASE8_PRETRAINING_AUDIT.md",
        "PHASE8_TRAINING_POLICY.md",
        "PHASE8_THRESHOLD_AND_CALIBRATION.md",
        "PHASE8_ERROR_ANALYSIS.md",
        "PHASE8_ROOT_CAUSE.md",
        "PHASE8_RECOVERY_DECISION.md",
        "PHASE8_VALIDATION.md",
        "PHASE8_GATE.md",
    }
    assert all((ARTIFACT_ROOT / name).is_file() for name in artifacts)
    assert all((REPORT_ROOT / name).is_file() for name in reports)


def test_stored_h0_and_h1_metrics_reproduce_exactly() -> None:
    evidence = _json("metric_reproduction.json")
    assert evidence["status"] == "PASS"
    assert evidence["H0"]["pass"] is True
    assert evidence["H1"]["pass"] is True
    assert evidence["H0"]["absolute_difference"] <= 1e-12
    assert evidence["H1"]["absolute_difference"] <= 1e-12
    assert abs(evidence["macro_f1_delta_h1_minus_h0"] + 0.02968350593583491) <= 1e-12


def test_endpoint_population_target_and_outer_folds_match() -> None:
    population = _json("target_population_diff.json")
    assert population["status"] == "IDENTICAL"
    assert population["record_identity_exact"] is True
    assert population["fold_assignment_identity_exact"] is True
    assert population["eligible_records"] == 15_378
    assert population["positive_count"] == 6_118
    assert population["negative_count"] == 9_260
    assert population["outer_group_overlap"] == 0
    assert population["fold_counts"] == {"0": 5120, "1": 5109, "2": 5149}


def test_feature_audit_exposes_score_authority_difference() -> None:
    features = pd.read_csv(ARTIFACT_ROOT / "feature_schema_diff.csv")
    score = features[
        features.H0_feature.fillna("").str.contains(
            "score", case=False, regex=False
        )
    ]
    assert not score.empty
    assert score.classification.str.contains(
        "DIFFERENT_AVAILABILITY"
    ).any()
    h0 = _json("h0_endpoint_profile.json")
    h1 = _json("h1_endpoint_profile.json")
    assert h0["features"]["aggregate_input_dimension"] == 49
    assert h1["features"]["aggregate_input_dimension"] == 165
    assert h0["feature_schema_hash"] != h1["feature_schema_hash"]


def test_threshold_is_diagnostic_only_and_not_primary_cause() -> None:
    evidence = _json("threshold_analysis.json")
    assert evidence["status"] == "THRESHOLD_NOT_PRIMARY_CAUSE"
    assert evidence["outer_labels_used_for_selection"] is False
    oracle = evidence["H1"]["diagnostic_outer_oracle"]
    assert oracle["scope"] == "DIAGNOSTIC_OUTER_ORACLE_NOT_FOR_SELECTION"
    gain = oracle["macro_f1"] - evidence["H1"]["registered"]["macro_f1"]
    assert 0.0 < gain < 0.001
    assert (
        evidence["H1"]["registered"]["pr_auc"]
        < evidence["H0"]["registered"]["pr_auc"]
    )


def test_pretraining_execution_has_historical_checkpoint_provenance() -> None:
    h0 = _json("h0_endpoint_profile.json")
    h1 = _json("h1_endpoint_profile.json")
    assert h0["training"]["pretraining_executed"] is True
    assert h0["training"]["pretraining"]["epochs"] == 5
    assert h0["model"]["checkpoint_count"] == 15
    assert len(h0["model"]["checkpoint_hashes"]) == 15
    assert h1["training"]["pretraining_executed"] is False


def test_early_warning_checksums_remain_unchanged() -> None:
    evidence = _json("early_warning_integrity.json")
    assert evidence["status"] == "PASS"
    assert evidence["modified"] is False
    for row in evidence["checks"]:
        assert _sha256(ROOT / row["path"]) == row["expected_sha256"]
        assert row["match"] is True


def test_phase8_is_static_and_selects_one_recovery_path() -> None:
    gate = _json("phase8_gate.json")
    recovery = _json("recovery_decision.json")
    assert gate["gate"] == "PASS"
    assert gate["training_performed"] is False
    assert gate["optuna_trials"] == 0
    assert gate["outer_evaluations"] == 0
    assert gate["outer_labels_used_for_selection"] is False
    assert recovery["selected_path"] == "R1"
    assert recovery["new_final_holdout_required"] is True
    assert recovery["training_launched"] is False
