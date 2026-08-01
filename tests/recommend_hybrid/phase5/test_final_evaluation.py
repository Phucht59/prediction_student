from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest


@pytest.fixture(scope="module")
def root() -> Path:
    return Path(__file__).resolve().parents[3]


@pytest.fixture(scope="module")
def metrics(root) -> dict:
    return json.loads(
        (root / "artifacts/recommend_hybrid/final/FINAL_METRICS.json").read_text(
            encoding="utf-8"
        )
    )


def test_locked_evaluation_scope(metrics):
    design = metrics["evaluation_design"]
    assert design["record_count"] == 260
    assert design["outcome_labels_used_for_policy_or_sampling"] is False
    assert design["pseudonymous_records_only"] is True


def test_uci_mat_evaluated(metrics):
    assert metrics["by_dataset"]["student_mat"]["record_count"] == 60
    assert all(f"student_mat:{stage}" in metrics["by_dataset_stage"] for stage in ("S0", "S1", "S2"))


def test_uci_por_evaluated(metrics):
    assert metrics["by_dataset"]["student_por"]["record_count"] == 60
    assert all(f"student_por:{stage}" in metrics["by_dataset_stage"] for stage in ("S0", "S1", "S2"))


def test_oulad_anchors_and_interstages_evaluated(metrics):
    stages = (
        "EARLY_20",
        "EARLY_35",
        "MIDDLE_50",
        "LATE_75",
        "FINAL_EVALUATION",
        "INTER_STAGE_25",
        "INTER_STAGE_36",
        "INTER_STAGE_63",
        "INTER_STAGE_76",
    )
    assert all(f"oulad:{stage}" in metrics["by_dataset_stage"] for stage in stages)


def test_safety_violations_are_zero(metrics):
    safety = (
        "post_cutoff_violations",
        "future_anchor_violations",
        "final_intervention_violations",
        "G3_usage",
        "sensitive_feature_violations",
        "missing_lineage_violations",
        "cross_dataset_policy_violations",
        "invalid_model_dataset_mapping",
    )
    assert all(metrics["violations"][name] == 0 for name in safety)


def test_constraint_violations_are_zero(metrics):
    names = (
        "action_cap_violations",
        "workload_violations",
        "duplicate_action_violations",
        "prerequisite_violations",
        "contraindication_violations",
        "unsupported_action_violations",
        "invalid_period_violations",
        "course_end_violations",
    )
    assert all(metrics["violations"][name] == 0 for name in names)


def test_evidence_and_explanation_complete(metrics):
    overall = metrics["overall"]
    assert overall["evidence_support_rate"] == 1.0
    assert overall["explanation_lineage_completeness"] == 1.0
    assert overall["unsupported_reason_rate"] == 0.0
    assert overall["missing_evidence_misuse_rate"] == 0.0


def test_determinism_and_robustness(metrics):
    assert metrics["reproducibility"]["deterministic_replay_rate"] == 1.0
    assert metrics["reproducibility"]["plan_hash_match_rate"] == 1.0
    assert metrics["robustness"]["status"] == "PASS"
    assert metrics["monotonicity_violation_count"] == 0


def test_ablation_is_complete(root):
    payload = json.loads(
        (root / "artifacts/recommend_hybrid/final/ABLATION_SUMMARY.json").read_text(
            encoding="utf-8"
        )
    )
    assert len(payload["results"]) == 4
    assert payload["official_variant"] == "D_OFFICIAL_FULL_POLICY"
    assert payload["outcome_labels_used"] is False


def test_bootstrap_is_student_level_and_fixed(root):
    payload = json.loads(
        (
            root
            / "artifacts/recommend_hybrid/final/BOOTSTRAP_CONFIDENCE_INTERVALS.json"
        ).read_text(encoding="utf-8")
    )
    assert payload["replicates"] == 1000
    assert payload["random_seed"] == 20260801
    assert payload["unit"] == "pseudonymous_student"
    assert len(payload["metrics"]) == 6


def test_claim_boundary_has_required_statuses(root):
    text = (root / "reports/recommend_hybrid/SCIENTIFIC_CLAIM_BOUNDARY.md").read_text(
        encoding="utf-8"
    )
    assert "Recommendations evidence-linked | SUPPORTED" in text
    assert "Recommendations improve grades | NOT_SUPPORTED" in text
    assert "Expert validated | NOT_SUPPORTED" in text
    assert "Causal effect established | NOT_SUPPORTED" in text


def test_model_card_report_and_stage_csv_present(root):
    model_card = (root / "docs/recommend_hybrid/MODEL_CARD.md").read_text(encoding="utf-8")
    thesis = (root / "docs/recommend_hybrid/THESIS_RECOMMENDATION_SYSTEM.md").read_text(
        encoding="utf-8"
    )
    with (root / "reports/recommend_hybrid/DATASET_STAGE_RESULTS.csv").open(
        encoding="utf-8", newline=""
    ) as stream:
        rows = list(csv.DictReader(stream))
    assert "deterministic evidence-based policy" in model_card
    assert "không phải neural ranker" in thesis
    assert len(rows) == 15
