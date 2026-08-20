"""5-fold CV firewall: outer fold 0 never used; architecture stays C0 CNN–BiLSTM."""
from __future__ import annotations

from pathlib import Path

from experiments.cv5.splits import cv5_partitions, outer0_ids
from experiments.imbalance.train_hybrid import make_config


def test_cv5_excludes_official_outer_fold0():
    blocked = outer0_ids("uci")
    assert len(blocked) > 0
    seen = set()
    for fold in range(5):
        fit, stop, valid, meta = cv5_partitions("uci", fold)
        used = set(fit) | set(stop) | set(valid)
        assert not (used & blocked)
        assert meta["outer0_excluded"] is True
        assert not (set(fit) & set(stop) or set(fit) & set(valid) or set(stop) & set(valid))
        seen.update(valid)
    assert len(seen) > 500


def test_cv5_keeps_c0_cnn_bilstm():
    cfg = make_config(8, 1, 5, {"dropout": 0.4, "entropy_floor_coefficient": 0.002})
    assert cfg.architecture_id == "C0"
    assert cfg.fusion == "softmax_3way"
    assert cfg.cnn_channels == 64
    assert cfg.bilstm_hidden == 128
    hybrid = (Path(__file__).resolve().parents[1] / "src" / "prediction" / "model" / "hybrid.py").read_text(encoding="utf-8")
    assert "ResidualCNNBranch" in hybrid
    assert "BiLSTMBranch" in hybrid
