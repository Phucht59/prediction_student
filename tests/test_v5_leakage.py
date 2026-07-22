import numpy as np

from src.studies.v5.common.protocol import ROOT, load_json_yaml
from src.studies.v5.common.uci_data import CONTEXT_NUMERIC, context_preprocessor, load_uci
from src.studies.oulad_v4.data import load_v4_data


def test_uci_preprocessor_statistics_come_only_from_training_rows():
    data = load_uci(ROOT / "data/raw/student-mat.csv", "student-mat")
    training = np.arange(200)
    transformer = context_preprocessor().fit(data.context.iloc[training])
    scaler = transformer.named_transformers_["numeric"].named_steps["scale"]
    expected = data.context.iloc[training][CONTEXT_NUMERIC].to_numpy(dtype=float).mean(axis=0)
    assert np.allclose(scaler.mean_, expected)


def test_oulad_v5_development_has_no_future_role_or_group_overlap():
    protocol = load_json_yaml(ROOT / "configs/oulad_v4_protocol.yaml")
    data = load_v4_data(ROOT / "data/processed/study_c_oulad", protocol)
    assert set(data.development_manifest.role) == {"historical_development"}
    for fold in range(3):
        train, validation = data.v2.outer_indices(fold)
        assert not (set(data.groups[train]) & set(data.groups[validation]))


def test_joint_uci_quasi_identity_is_stable_across_subject_files():
    mat = load_uci(ROOT / "data/raw/student-mat.csv", "student-mat")
    por = load_uci(ROOT / "data/raw/student-por.csv", "student-por")
    assert len(set(mat.quasi_groups) & set(por.quasi_groups)) > 0
    assert not np.array_equal(mat.record_ids[:10], por.record_ids[:10])
