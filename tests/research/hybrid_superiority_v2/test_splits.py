"""Group-disjoint split firewall tests. Skips if UCI cache is missing."""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from experiments.hybrid_superiority_v2.paths import CACHE_DIR
from experiments.hybrid_superiority_v2.protocol import DEVELOPMENT_OUTER_FOLD


pytestmark = pytest.mark.skipif(not (CACHE_DIR / "uci" / "manifest.json").exists(), reason="uci cache missing")


def test_group_disjointness_outer_and_inner():
    from experiments.hybrid_superiority_v2.data import inner_partitions, outer_test_ids

    blocked = outer_test_ids("uci")
    for fold in range(3):
        fit, stop, valid = inner_partitions("uci", fold)
        assert not (set(fit) & blocked)
        assert not (set(stop) & blocked)
        assert not (set(valid) & blocked)
        assert not (set(fit) & set(stop))
        assert not (set(fit) & set(valid))
        assert not (set(stop) & set(valid))
    outer = pd.read_parquet(CACHE_DIR / "splits" / "uci_outer.parquet")
    ctx = pd.read_parquet(CACHE_DIR / "uci" / "context.parquet")
    merged = outer.merge(ctx[["record_id", "group_id"]], on="record_id", suffixes=("", "_ctx"))
    folds = sorted(merged.outer_fold.unique())
    for i in folds:
        for j in folds:
            if i >= j:
                continue
            gi = set(merged.loc[merged.outer_fold == i, "group_id"].astype(str))
            gj = set(merged.loc[merged.outer_fold == j, "group_id"].astype(str))
            assert not (gi & gj)


def test_preprocessor_fit_ids_recorded():
    from experiments.hybrid_superiority_v2.data import inner_partitions, scale_views

    fit, stop, valid = inner_partitions("uci", 0)
    prepared = scale_views("uci", fit)
    assert prepared.static_dim > 0
    assert prepared.feature_contract["g1_g2_in_hybrid_static"] is False
    assert prepared.feature_contract["fit_only_scaling"] is True
    # G1/G2 must not appear in Hybrid static map dimensionality leak via names
    assert "G1" not in prepared.numeric
    assert "G2" not in prepared.numeric
