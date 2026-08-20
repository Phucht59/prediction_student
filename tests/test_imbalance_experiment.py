"""Isolation and leakage tests for the SMOTE/ADASYN experiment."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from experiments.imbalance.integrity import compare, snapshot
from experiments.imbalance.samplers import PackedBatch, pack_features, resample_train, unpack_features
from experiments.imbalance.train_hybrid import make_config
from src.prediction.model import Hybrid

ROOT = Path(__file__).resolve().parents[1]


def _batch(n=40, t=2, ds=8, da=5, dt=1, p_pos=0.25, seed=0) -> PackedBatch:
    rng = np.random.default_rng(seed)
    y = np.zeros(n, np.int64)
    y[: max(4, int(n * p_pos))] = 1
    rng.shuffle(y)
    mask = np.zeros((n, t), dtype=bool)
    mask[:, 0] = True
    temporal = rng.normal(size=(n, t, dt)).astype(np.float32) * mask[..., None]
    return PackedBatch(
        static=rng.normal(size=(n, ds)).astype(np.float32),
        temporal=temporal,
        temporal_mask=mask,
        lengths=mask.sum(1).astype(np.int64),
        aggregate=rng.normal(size=(n, da)).astype(np.float32),
        aggregate_available=np.ones(n, np.int8),
        progress=np.full(n, 0.5, np.float32),
        target=y,
        record_id=np.asarray([f"r{i}" for i in range(n)], dtype=object),
    )


def test_sampler_only_sees_training_ids():
    train = _batch(seed=1)
    valid = _batch(n=20, seed=2)
    valid_hash = hashlib.sha256(valid.static.tobytes() + valid.target.tobytes()).hexdigest()
    packed = pack_features(train)
    x_new, y_new, audit = resample_train(packed, train.target, "smote", random_state=0)
    assert audit["fit_on"] == "train_only"
    assert audit["n_train_after"] >= audit["n_train_before"]
    after_valid = hashlib.sha256(valid.static.tobytes() + valid.target.tobytes()).hexdigest()
    assert after_valid == valid_hash
    assert x_new.shape[1] == packed.shape[1]


def test_validation_and_test_unchanged_under_adasyn():
    valid = _batch(n=16, seed=3)
    test = _batch(n=16, seed=4)
    before = (valid.static.copy(), test.static.copy(), valid.target.copy(), test.target.copy())
    resample_train(pack_features(_batch(seed=5)), _batch(seed=5).target, "adasyn", random_state=1)
    assert np.array_equal(valid.static, before[0])
    assert np.array_equal(test.static, before[1])
    assert np.array_equal(valid.target, before[2])
    assert np.array_equal(test.target, before[3])


def test_output_dimensions_remain_valid():
    train = _batch()
    packed = pack_features(train)
    x_new, y_new, _ = resample_train(packed, train.target, "smote", random_state=0)
    out = unpack_features(
        x_new,
        static_dim=train.static.shape[1],
        aggregate_dim=train.aggregate.shape[1],
        timesteps=train.temporal.shape[1],
        temporal_dim=train.temporal.shape[2],
        target=y_new,
        stage_mask_template=np.array([True, False]),
    )
    assert out.static.shape[1] == train.static.shape[1]
    assert out.temporal.shape[1:] == train.temporal.shape[1:]
    assert (out.lengths == out.temporal_mask.sum(1)).all()


def test_hybrid_availability_s0_zero_temporal_mass():
    cfg = make_config(8, 1, 5, {"dropout": 0.2, "entropy_floor_coefficient": 0.002})
    model = Hybrid(cfg)
    model.eval()
    n, t = 4, 2
    static = torch.randn(n, 8)
    temporal = torch.zeros(n, t, 1)
    mask = torch.zeros(n, t, dtype=torch.bool)
    lengths = torch.zeros(n, dtype=torch.long)
    aggregate = torch.randn(n, 5)
    agg_ok = torch.ones(n)
    progress = torch.zeros(n)
    with torch.no_grad():
        logits = model(static, temporal, mask, lengths, aggregate, agg_ok, progress)
    assert torch.isfinite(logits).all()
    w = model.last_diagnostics["gate_weights"]
    assert float(w[:, 1].max()) < 1e-5
    assert float(w[:, 2].max()) < 1e-5
    assert float(w[:, 0].min()) > 0.99


def test_oulad_cutoff_rule_still_in_production_code():
    text = (ROOT / "src" / "prediction" / "data" / "oulad_features.py").read_text(encoding="utf-8")
    assert "event_time < cutoff" in text


def test_production_prediction_path_unchanged_helper():
    before = snapshot()
    after = snapshot()
    payload = compare(before, after)
    assert payload["MODEL_CHANGED"] is False
    assert payload["HPO_PERFORMED"] is False
    assert payload["OUTER_OPENED"] is False


def test_experiment_is_isolated_from_src_prediction():
    hybrid = (ROOT / "src" / "prediction" / "model" / "hybrid.py").read_text(encoding="utf-8")
    assert "SMOTE" not in hybrid
    assert "ADASYN" not in hybrid
    pred_init = (ROOT / "src" / "prediction" / "__init__.py").read_text(encoding="utf-8")
    assert "imbalance" not in pred_init
    assert (ROOT / "experiments" / "imbalance" / "samplers.py").is_file()


def test_isolated_trainer_uses_frozen_c0_topology():
    payload = json.loads((ROOT / "artifacts" / "prediction" / "final" / "TRAINING_CONFIG.json").read_text(encoding="utf-8"))
    assert payload["outer_test_used"] is False
    cfg = make_config(10, 1, 5, payload["uci"])
    assert cfg.architecture_id == "C0"
    assert cfg.d_fuse == 128
    assert cfg.cnn_channels == 64
    assert cfg.bilstm_hidden == 128
    assert cfg.fusion == "softmax_3way"
    assert cfg.dropout == float(payload["uci"]["dropout"])


def test_uci_recovered_partitions_match_official_oof_valid():
    from experiments.imbalance.data_build import partitions

    expected = {0: 284, 1: 272, 2: 274}
    seen = set()
    for fold, n_valid in expected.items():
        fit, stop, valid, meta = partitions("uci", fold)
        assert len(valid) == n_valid
        assert not (set(fit) & set(stop) or set(fit) & set(valid) or set(stop) & set(valid))
        assert meta["split_source"] in {"frozen_kltn_parquet", "recovered_from_official_oof_valid"}
        seen.update(valid)
        assert not any(i.startswith("synth:") for i in fit + stop + valid)
    assert len(seen) == sum(expected.values())


def test_control_sampler_is_passthrough():
    train = _batch(seed=11)
    packed = pack_features(train)
    x_new, y_new, audit = resample_train(packed, train.target, "control", random_state=0)
    assert audit["n_train_after"] == audit["n_train_before"]
    assert np.array_equal(x_new, packed)
    assert np.array_equal(y_new, train.target)


def test_smote_and_adasyn_are_separate_modes():
    train = _batch(seed=9)
    packed = pack_features(train)
    xs, ys, a_s = resample_train(packed, train.target, "smote", random_state=0)
    xa, ya, a_a = resample_train(packed, train.target, "adasyn", random_state=0)
    assert a_s["sampler"] == "smote"
    assert a_a["sampler"] == "adasyn"
    assert xs.shape[1] == xa.shape[1] == packed.shape[1]
    assert set(np.unique(ys)) <= {0, 1}
    assert set(np.unique(ya)) <= {0, 1}
