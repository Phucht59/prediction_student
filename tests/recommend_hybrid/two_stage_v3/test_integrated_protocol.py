from __future__ import annotations

from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[3]


def protocol() -> dict:
    return yaml.safe_load(
        (ROOT / "configs/recommend_hybrid/two_stage_v3_protocol.yaml").read_text(
            encoding="utf-8"
        )
    )


def test_backbone_is_frozen_and_external_rankers_are_forbidden() -> None:
    payload = protocol()
    assert payload["architecture"]["frozen_prediction_backbone"]["trainable"] is False
    assert payload["architecture"]["integrated_heads"]["external_ml_ranker"] is False
    assert "xgboost" in payload["architecture"]["prohibited_external_models"]
    assert "lambdamart" in payload["architecture"]["prohibited_external_models"]


def test_labels_and_eighty_percent_gate_are_unchanged() -> None:
    payload = protocol()
    assert payload["targets"]["labels_changed_from_hybrid_only_execution"] is False
    assert payload["targets"]["future_labels_used_at_runtime"] is False
    assert payload["release_gates"]["end_to_end_precision_at_1_minimum"] == 0.80
    assert payload["release_gates"]["positive_group_coverage_minimum"] == 0.50


def test_difficult_stage_and_action_remain_reportable() -> None:
    payload = protocol()
    assert "EARLY_20" in payload["evaluation"]["stages"]
    assert payload["action_registry"]["assessment_completion"][
        "retained_in_training_and_full_reporting"
    ] is True
    assert payload["action_registry"]["assessment_completion"][
        "may_not_be_silently_removed"
    ] is True


def test_registered_search_is_bounded_before_training() -> None:
    payload = protocol()
    assert payload["head_training"]["registered_trial_count"] == 12
    assert payload["selection"]["threshold_source"] == "inner_oof_only"
    assert payload["head_training"]["fixed_epoch_source"] == "median_inner_best_epoch"
