from __future__ import annotations

from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[3]


def protocol() -> dict:
    return yaml.safe_load(
        (ROOT / "configs/recommend_hybrid/two_stage_v4_protocol.yaml").read_text(
            encoding="utf-8"
        )
    )


def test_backbone_and_release_gates_are_unchanged() -> None:
    payload = protocol()
    assert payload["architecture"]["frozen_prediction_backbone"]["parameter_count"] == 160492
    assert payload["architecture"]["frozen_prediction_backbone"]["trainable"] is False
    assert payload["release_gates"]["end_to_end_precision_at_1_minimum"] == 0.80
    assert payload["release_gates"]["positive_group_coverage_minimum"] == 0.50


def test_candidate_binary_supervision_includes_negative_groups() -> None:
    payload = protocol()
    assert payload["targets"]["candidate_binary_population"] == "all_groups_all_valid_candidates"
    assert payload["scientific_change_from_v3"]["labels_changed"] is False
    assert payload["scientific_change_from_v3"]["frozen_embeddings_changed"] is False


def test_external_ranker_remains_forbidden() -> None:
    payload = protocol()
    assert payload["architecture"]["integrated_heads"]["external_ml_ranker"] is False
    prohibited = set(payload["architecture"]["prohibited_external_models"])
    assert {"xgboost", "lightgbm", "lambdamart", "logistic_regression"} <= prohibited


def test_registered_configs_are_fixed_before_training() -> None:
    payload = protocol()
    configs = payload["head_training"]["registered_configs"]
    assert len(configs) == 12
    assert payload["status"] == "FROZEN_BEFORE_V4_HEAD_TRAINING"
