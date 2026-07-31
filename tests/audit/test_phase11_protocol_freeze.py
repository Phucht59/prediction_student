from __future__ import annotations

import json
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "artifacts" / "canonical_v3"


def test_policy_is_strict_and_frozen_before_benchmark() -> None:
    policy = yaml.safe_load(
        (ROOT / "configs/canonical_v3/oulad_information_policy.yaml").read_text(
            encoding="utf-8"
        )
    )
    assert policy["policy"] == "STRICT_REAL_TIME"
    assert policy["score_policy"]["score_values"] == "EXCLUDED"
    assert policy["status"] == "FROZEN_BEFORE_BENCHMARK"


def test_information_monotonicity_and_final_superset() -> None:
    audit = json.loads((OUT / "oulad_feature_monotonicity.json").read_text(encoding="utf-8"))
    assert audit["status"] == "PASS"
    assert all(audit["relations"].values())
    assert audit["75_only_features"] == []
    assert audit["score_values_excluded_all_stages"] is True


def test_old_endpoint_was_not_post_75_final() -> None:
    audit = json.loads(
        (OUT / "old_75_vs_endpoint_protocol_audit.json").read_text(encoding="utf-8")
    )
    assert audit["same_score_policy"] is True
    assert audit["endpoint_cutoff_fraction"] == 0.5
    assert audit["directly_comparable_as_75_vs_final"] is False


def test_architecture_families_and_compute_policy_are_frozen() -> None:
    protocol = yaml.safe_load(
        (ROOT / "configs/canonical_v3/benchmark_protocol.yaml").read_text(encoding="utf-8")
    )
    freeze = json.loads((OUT / "CANONICAL_BENCHMARK_FREEZE.json").read_text(encoding="utf-8"))
    assert protocol["uci"]["hybrid"]["architecture_id"] == "CNN_BILSTM_UCI_CANONICAL"
    assert protocol["oulad"]["hybrid"]["architecture_id"] == "H1_TABULAR_RESIDUAL_EXPERT"
    assert freeze["oulad_parameter_count"] == 160492
    assert freeze["architecture_search"] is False
    assert freeze["new_benchmark_metrics_observed_before_freeze"] is False
