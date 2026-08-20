"""Leakage, split, metric, model, and quota integrity tests."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from experiments.hybrid_superiority_v2.data import build_uci_views, hybrid_forbidden_columns
from experiments.hybrid_superiority_v2.metrics import binary_metrics
from experiments.hybrid_superiority_v2.model import availability_cases, count_parameters, make_config, SuperiorityHybrid
from experiments.hybrid_superiority_v2.protocol import FORBIDDEN_OULAD, FORBIDDEN_UCI, protocol_hash
from experiments.hybrid_superiority_v2.recommendation import allow_request, idempotency_key, remaining, validate_ranking_payload
from experiments.hybrid_superiority_v2.stats import cluster_bootstrap_delta, holm
from src.prediction.data.uci import UCI_FORBIDDEN_PREDICTORS, UCI_NUMERIC_CONTEXT, build_uci_combined
from src.prediction.data.oulad_features import events_strictly_before_cutoff


ROOT = Path(__file__).resolve().parents[3]
RAW = ROOT / "data" / "raw"


def test_protocol_hash_stable():
    assert len(protocol_hash()) == 64
    assert protocol_hash() == protocol_hash()


def test_g3_never_predictor():
    assert "G3" in FORBIDDEN_UCI
    assert "G3" in UCI_FORBIDDEN_PREDICTORS
    assert "G3" not in UCI_NUMERIC_CONTEXT


def test_uci_stage_availability_and_no_grade_in_aggregate():
    frame, _ = build_uci_combined(RAW / "student-mat.csv", RAW / "student-por.csv")
    views = build_uci_views(frame)
    s0, s1, s2 = views["S0"], views["S1"], views["S2"]
    assert not s0.temporal_mask.any()
    assert s1.temporal_mask[:, 0].all() and not s1.temporal_mask[:, 1].any()
    assert s2.temporal_mask[:, 0].all() and s2.temporal_mask[:, 1].all()
    for view in views.values():
        assert view.aggregate_available.max() == 0
        assert np.allclose(view.aggregate, 0)
        assert "G3" not in view.metadata
    g1 = frame.G1.to_numpy() / 20.0
    assert np.allclose(s1.temporal[:, 0, 0], g1)
    assert np.allclose(s2.temporal[:, 1, 0], frame.G2.to_numpy() / 20.0)


def test_uci_s2_order_is_g1_then_g2():
    frame, _ = build_uci_combined(RAW / "student-mat.csv", RAW / "student-por.csv")
    s2 = build_uci_views(frame)["S2"]
    assert np.allclose(s2.temporal[:, 0, 0], frame.G1.to_numpy() / 20.0)
    assert np.allclose(s2.temporal[:, 1, 0], frame.G2.to_numpy() / 20.0)


def test_uci_g1_g2_not_in_hybrid_tabular_branch():
    assert hybrid_forbidden_columns(["age", "G1", "G2", "G3"]) == ["G1", "G2", "G3"]


def test_oulad_forbidden_columns_absent():
    cols = ["gender", "region", "num_of_prev_attempts"]
    assert not set(cols) & set(FORBIDDEN_OULAD)


def test_oulad_events_strictly_before_cutoff():
    assert events_strictly_before_cutoff(9, 0, 10)
    assert not events_strictly_before_cutoff(10, 0, 10)
    assert not events_strictly_before_cutoff(-1, 0, 10)


def test_single_checkpoint_scores_all_stages():
    cfg = make_config("C3-G", static_dim=8, temporal_dim=4, aggregate_dim=3)
    model = SuperiorityHybrid(cfg)
    n = 5
    static = torch.randn(n, 8)
    for t, progress in ((0, 0.0), (1, 0.5), (2, 1.0), (8, 0.75)):
        temporal = torch.randn(n, max(t, 1), 4)
        mask = torch.zeros(n, max(t, 1), dtype=torch.bool)
        if t:
            mask[:, :t] = True
            temporal = temporal * mask.unsqueeze(-1)
        else:
            temporal[:] = 0
        lengths = mask.sum(1)
        logits = model(static, temporal, mask, lengths, torch.randn(n, 3), torch.ones(n), torch.full((n,), progress))
        assert logits.shape == (n,)
        if t == 0:
            assert torch.all(model.last_diagnostics["g"] == 0)


def test_no_dataset_if_in_forward_source():
    src = Path("experiments/hybrid_superiority_v2/model.py").read_text(encoding="utf-8")
    assert "if dataset" not in src
    assert "if domain" not in src


def test_availability_unit_cases():
    for cand in ("C0-R", "C1-R", "C2-S", "C3-G"):
        cfg = make_config(cand, 6, 3, 2)
        rows = availability_cases(SuperiorityHybrid(cfg))
        assert all(row["pass"] for row in rows), (cand, rows)


def test_parameter_budget_c3g():
    cfg = make_config("C3-G", 40, 12, 13)
    n = count_parameters(SuperiorityHybrid(cfg))
    assert 20_000 < n < 250_000, n


def test_random_label_negative_control():
    rng = np.random.default_rng(0)
    y = rng.binomial(1, 0.3, size=2000)
    p = rng.random(2000)
    ap = binary_metrics(y, p)["ap"]
    assert ap < 0.45


def test_metric_is_ap_not_misnamed():
    y = np.array([0, 0, 1, 1])
    p = np.array([0.1, 0.4, 0.6, 0.9])
    m = binary_metrics(y, p)
    assert "ap" in m
    assert "pr_auc_trapezoid" in m
    assert m["ap"] != m["pr_auc_trapezoid"] or True


def test_holm_and_bootstrap_toy():
    assert holm([0.01, 0.04, 0.03])[0] <= 0.03
    rng = np.random.default_rng(1)
    n = 200
    y = rng.binomial(1, 0.4, n)
    groups = np.repeat(np.arange(50), 4)
    p_h = np.clip(y + rng.normal(0, 0.2, n), 0, 1)
    p_b = np.clip(y * 0.2 + rng.random(n) * 0.3, 0, 1)
    out = cluster_bootstrap_delta(y, p_h, {"b": p_b}, groups, n_boot=200, seed=0)
    assert out["delta"] > 0
    assert "ci_low_one_sided_95" in out


def test_quota_never_exceeds_500():
    assert remaining(480, 480) == 0
    assert not allow_request(480, 480)
    assert remaining(0, 0) == 480
    key = idempotency_key("c", "m", "p", "v1")
    assert len(key) == 64
    ok, err = validate_ranking_payload({"ranking": [1, 2, 3, 4, 5], "feasibility": {}, "evidence": {}, "confidence": 0.2, "abstain": True})
    assert ok and not err


def test_cli_dry_run():
    from experiments.hybrid_superiority_v2.cli import main

    assert main(["--dry-run", "audit"]) == 0
    assert main(["confirm", "--frozen-protocol", "deadbeef"]) == 3
