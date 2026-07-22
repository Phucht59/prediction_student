import numpy as np

from src.studies.v5.common.protocol import ROOT
from src.studies.v5.common.uci_data import CONTEXT_FEATURES, load_uci


def test_uci_v5_uses_g1_g2_sequence_and_never_g3_as_input():
    data = load_uci(ROOT / "data/raw/student-mat.csv", "student-mat")
    assert data.sequence.shape == (395, 2, 1)
    assert list(data.context.columns) == CONTEXT_FEATURES
    assert "G3" not in data.context.columns
    assert np.array_equal(data.sequence[:, :, 0], data.frame[["G1", "G2"]].to_numpy())


def test_uci_targets_follow_frozen_three_class_boundaries():
    data = load_uci(ROOT / "data/raw/student-por.csv", "student-por")
    assert np.array_equal(data.target, np.where(data.raw_g3 <= 9, 0, np.where(data.raw_g3 <= 14, 1, 2)))


def test_uci_source_identity_is_stable_and_unique():
    first = load_uci(ROOT / "data/raw/student-mat.csv", "student-mat")
    second = load_uci(ROOT / "data/raw/student-mat.csv", "student-mat")
    assert len(set(first.record_ids)) == len(first.record_ids)
    assert np.array_equal(first.record_ids, second.record_ids)

