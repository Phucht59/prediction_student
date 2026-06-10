from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch

from src.paper_replication.pipeline import (
    DATASETS,
    PaperCNNBiLSTM,
    add_targets,
    prepare_dataset,
)
from src.paper_replication.v18_strict_validation import prepare_strict_split, strict_student_cases


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_student_g3_is_mapped_to_five_paper_classes():
    spec = DATASETS["student-mat"]
    raw = pd.DataFrame({"G1": [1, 5, 9, 13, 17], "G2": [2, 6, 10, 14, 18], "G3": [0, 5, 9, 13, 20]})
    mapped = add_targets(raw, spec)

    assert mapped["target_class"].tolist() == [0, 1, 2, 3, 4]
    assert mapped["target_class_name"].tolist() == ["0-4", "5-8", "9-12", "13-16", "17-20"]


def test_paper_cnn_bilstm_forward_for_student_and_xapi_shapes():
    student_model = PaperCNNBiLSTM(input_dim=2, num_classes=5, conv_channels=8, conv_blocks=1, bilstm_hidden=6)
    xapi_model = PaperCNNBiLSTM(input_dim=8, num_classes=3, conv_channels=8, conv_blocks=2, bilstm_hidden=6)

    assert student_model(torch.randn(4, 2)).shape == (4, 5)
    assert xapi_model(torch.randn(4, 8)).shape == (4, 3)


def test_paper_prepare_uses_reported_features(tmp_path):
    metadata = prepare_dataset("student-mat", seed=42)

    assert metadata["target"] == "G3 five-class bins"
    assert metadata["resolved_feature_columns"] == ["G1", "G2"]
    assert metadata["n_features_processed"] == 2
    assert set(metadata["class_distribution_total"]) == {"0", "1", "2", "3", "4"}


def test_postgres_schema_contains_paper_tables():
    schema = (PROJECT_ROOT / "database" / "schema.sql").read_text(encoding="utf-8").lower()
    for table_name in ("student_grades", "paper_runs", "paper_predictions"):
        assert f"create table if not exists {table_name}" in schema


def test_strict_split_uses_disjoint_train_validation_test_sets():
    raw = pd.read_csv(PROJECT_ROOT / "data" / "raw" / "student-mat.csv", sep=";")
    split = prepare_strict_split("student-mat", raw, strict_student_cases()[0], seed=42)

    train = set(split.train_indices.tolist())
    val = set(split.val_indices.tolist())
    test = set(split.test_indices.tolist())

    assert train.isdisjoint(val)
    assert train.isdisjoint(test)
    assert val.isdisjoint(test)
    assert len(train) + len(val) + len(test) == len(raw)
    assert abs(len(train) / len(raw) - 0.70) < 0.02
    assert abs(len(val) / len(raw) - 0.15) < 0.02
    assert abs(len(test) / len(raw) - 0.15) < 0.02
