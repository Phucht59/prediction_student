from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.studies.student_por.data import encode_g3, load_student_csv, overlap_membership
from src.studies.student_por.evaluation import summary_metrics, validate_probabilities
from src.studies.student_por.models import neural_configs


ROOT = Path(__file__).resolve().parents[1]


def test_student_por_source_and_target_contract():
    frame = load_student_csv(ROOT / "data" / "raw" / "student-por.csv", "student-por")
    assert len(frame) == 649
    assert frame["source_record_id"].nunique() == 649
    assert frame["G3"].value_counts().sort_index().to_dict() == {0: 100, 1: 418, 2: 131}
    assert encode_g3(pd.Series([0, 9, 10, 14, 15, 20])).tolist() == [0, 0, 1, 1, 2, 2]


def test_overlap_audit_preserves_ambiguity():
    mat = load_student_csv(ROOT / "data" / "raw" / "student-mat.csv", "student-mat")
    por = load_student_csv(ROOT / "data" / "raw" / "student-por.csv", "student-por")
    membership, audit = overlap_membership(mat, por)
    assert audit["standard_inner_join_rows"] == 382
    assert audit["shared_unique_quasi_identity_keys"] == 366
    assert audit["unambiguous_one_to_one_keys"] == 358
    assert audit["ambiguous_shared_keys"] == 8
    assert membership.value_counts()["conservative_unmatched"] == 275


def test_probability_and_metric_recomputation_contract():
    probabilities = np.array([[0.8, 0.1, 0.1], [0.1, 0.8, 0.1], [0.1, 0.2, 0.7]])
    validate_probabilities(probabilities)
    frame = pd.DataFrame({"true_label": [0, 1, 2], "predicted_label": [0, 1, 2], "prob_low": probabilities[:, 0], "prob_medium": probabilities[:, 1], "prob_high": probabilities[:, 2]})
    metrics, classes = summary_metrics(frame)
    assert metrics["macro_f1"] == 1.0
    assert len(classes) == 3


def test_all_neural_candidates_resolve_without_batchnorm_or_resampling():
    for candidate in ["B-M0", "B-C0", "B-L1", "B-H1", "B-O0"]:
        configs = neural_configs(candidate)
        assert configs
        for config in configs:
            assert config["fixed_constants"]["batch_norm_allowed"] is False
            assert config["drop_last_train"] is False
            assert config["oversample_method"] == "none"
            assert config["parameter_count"] <= 5000
