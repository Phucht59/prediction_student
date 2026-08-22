"""C4-STAR v2.1 integrity: leakage, residual-zero, protocol freeze, splits untouched."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from experiments.c4_star.model import C4STAR, make_c4_config, zero_residual_matches_anchor, count_parameters
from experiments.c4_star.objective import constrained_J
from experiments.c4_star.protocol import PARENT_PROTOCOL_ID, PROTOCOL_ID, protocol_hash, protocol_payload
from experiments.hybrid_superiority_v2.data import hybrid_forbidden_columns
from experiments.hybrid_superiority_v2.protocol import FORBIDDEN_UCI, protocol_hash as parent_hash
from src.prediction.data.oulad_features import events_strictly_before_cutoff


def test_protocol_hashes_stable_and_distinct():
    assert PROTOCOL_ID == "c4_star_v2.1"
    assert PARENT_PROTOCOL_ID == "hybrid_superiority_v2.0"
    assert protocol_hash() == protocol_hash()
    assert len(protocol_hash()) == 64
    assert protocol_hash() != parent_hash()
    assert protocol_payload()["outer_splits_regenerated"] is False
    assert protocol_payload()["outer_test_used_for_selection"] is False


def test_g3_still_forbidden():
    assert "G3" in FORBIDDEN_UCI
    assert hybrid_forbidden_columns(["age", "G1", "G2", "G3"]) == ["G1", "G2", "G3"]


def test_oulad_cutoff_strict():
    assert events_strictly_before_cutoff(9, 0, 10)
    assert not events_strictly_before_cutoff(10, 0, 10)


def test_zero_residual_reproduces_anchor():
    cfg = make_c4_config(8, 4, 3, mechanism="M3", initial_alpha=0.05)
    model = C4STAR(cfg)
    n, t = 6, 5
    batch = dict(
        static=torch.randn(n, 8),
        temporal=torch.randn(n, t, 4),
        temporal_mask=torch.ones(n, t, dtype=torch.bool),
        lengths=torch.full((n,), t),
        aggregate=torch.randn(n, 3),
        aggregate_available=torch.ones(n, dtype=torch.bool),
        progress=torch.rand(n),
    )
    assert zero_residual_matches_anchor(model, batch)


def test_param_count_in_band():
    cfg = make_c4_config(49, 12, 13, d_fuse=64, cnn_channels=32, bilstm_hidden=48)
    model = C4STAR(cfg)
    n = count_parameters(model)
    assert 50_000 <= n <= 400_000


def test_constrained_j_penalizes_warm_loss():
    ceiling = {"S1": 0.77, "S2": 0.91, "S0": 0.50}
    win = constrained_J({"S1": 0.80, "S2": 0.93, "S0": 0.48}, ceiling, "uci")
    lose = constrained_J({"S1": 0.76, "S2": 0.90, "S0": 0.48}, ceiling, "uci")
    assert lose["n_warm_loss"] >= 1
    assert lose["J"] < win["J"]


def test_parent_splits_not_regenerated():
    root = Path("artifacts/research/hybrid_superiority_v2/cache/splits")
    if not (root / "uci_split_lock.json").exists():
        pytest.skip("cache missing")
    import json

    uci = json.loads((root / "uci_split_lock.json").read_text(encoding="utf-8"))
    oulad = json.loads((root / "oulad_split_lock.json").read_text(encoding="utf-8"))
    assert uci["outer_sha256"] == "4bf33619395c360442531d396575f42d3dae99e646da3d6418bf1070e8228d0b"
    assert oulad["outer_sha256"] == "8ad606ebe805cc0f6c9e742823f8db56122a1d8d6e932caf6d2cf36de09bcbec"
    assert uci["outer_test_excluded_from_development"] is True


def test_kd_bce_safe_under_autocast():
    from experiments.c4_star.losses import kd_bce

    if not torch.cuda.is_available():
        pytest.skip("cuda")
    logits = torch.randn(8, device="cuda", dtype=torch.float16)
    teacher = torch.rand(8, device="cuda")
    with torch.autocast("cuda", dtype=torch.float16):
        loss = kd_bce(logits, teacher, 2.0)
    assert torch.isfinite(loss.float())


def test_ssl_zero_when_short_prefix():
    from experiments.c4_star.losses import ssl_reconstruct

    pred = torch.randn(4, 2, 3)
    target = torch.randn(4, 2, 3)
    mask = torch.zeros(4, 2, dtype=torch.bool)
    assert float(ssl_reconstruct(pred, target, mask)) == 0.0
