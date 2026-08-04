from __future__ import annotations

from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[3]


def protocol() -> dict:
    return yaml.safe_load(
        (ROOT / "configs/recommend_hybrid/hybrid_only_final_protocol.yaml").read_text(
            encoding="utf-8"
        )
    )


def test_only_frozen_hybrid_is_allowed_as_learned_model() -> None:
    payload = protocol()
    assert payload["architecture"]["learned_models_allowed"] == [
        "frozen_residual_cnn_bilstm"
    ]
    assert payload["architecture"]["learned_recommendation_model_allowed"] is False


def test_eighty_percent_gate_and_coverage_are_frozen() -> None:
    payload = protocol()
    assert payload["release_gates"]["top1_precision_minimum"] == 0.80
    assert payload["release_gates"]["actionable_coverage_minimum"] == 0.50
    assert payload["release_gates"]["top1_precision_bootstrap_lower_minimum"] == 0.78


def test_future_labels_are_evaluation_only() -> None:
    payload = protocol()
    assert payload["silver_label"]["labels_used_at_runtime"] is False
    assert payload["architecture"]["future_labels_role"] == "held_out_silver_evaluation_only"


def test_runtime_filtering_order_is_preregistered() -> None:
    payload = protocol()
    assert payload["search_strategy"]["candidate_filter_before_ranking"] is True
    assert payload["search_strategy"]["weight_shortlist"] == 12


def test_policy_only_actions_are_not_scored_as_silver_actions() -> None:
    excluded = set(protocol()["action_mapping"]["excluded_policy_only_actions"])
    assert excluded == {
        "INSTRUCTOR_CONTACT",
        "ADVISOR_ESCALATION",
        "DIAGNOSTIC_CHECK",
        "PROGRESS_MONITORING",
    }
