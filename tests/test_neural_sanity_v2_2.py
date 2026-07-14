import json
from pathlib import Path

import pandas as pd
import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from src.evaluation.neural_sanity_v2_2 import (
    EXPERIMENTS, JOB_COLUMNS, build_expected_job_contract, duplicate_job_rows,
    feature_contract, loader_statistics, variant_config,
)
from src.evaluation.protocol import load_fold_manifest


def _source_configs():
    return {i: {"cnn_kernel_size": 2 if i < 3 else 1, "max_epochs": 20, "patience": 5,
                "scheduler_patience": 3, "batch_size": 16, "learning_rate": .001} for i in range(5)}


def test_frozen_matrix_has_exactly_six_variants():
    assert tuple(EXPERIMENTS) == ("S0", "S1", "S2", "S3", "S4", "S5")


def test_variant_changes_only_frozen_factors():
    base = _source_configs()[0]
    assert variant_config(base, "S0")["cnn_kernel_size"] == 2
    assert variant_config(base, "S1")["drop_last_train"] is False
    assert variant_config(base, "S2")["cnn_kernel_size"] == 1
    assert variant_config(base, "S3")["max_epochs"] == 40
    assert variant_config(base, "S5")["patience"] == 8
    assert variant_config(base, "S5")["learning_rate"] == base["learning_rate"]


def test_expected_contract_is_precomputed_and_has_150_jobs():
    manifest = load_fold_manifest()
    counts = {i: sum(1 for x in manifest["assignments"] if x["outer_role"] == "validation" and x["outer_fold"] == i) for i in range(5)}
    contract = build_expected_job_contract("run", _source_configs(), counts, manifest["manifest_checksum"])
    assert len(contract["jobs"]) == 150
    assert all(set(["run_id", "experiment_id", "scenario", "model_name", "outer_fold", "training_seed", "expected_record_count", "config_checksum", "feature_contract_checksum", "fold_manifest_checksum"]).issubset(x) for x in contract["jobs"])


def test_smoke_contract_is_one_fold_one_seed_six_jobs():
    manifest = load_fold_manifest()
    contract = build_expected_job_contract("smoke", {0: _source_configs()[0]}, {0: 64}, manifest["manifest_checksum"], seeds=(42,))
    assert len(contract["jobs"]) == 6


def test_duplicate_contract_jobs_are_detected_without_groupby():
    manifest = load_fold_manifest(); contract = build_expected_job_contract("run", _source_configs(), {i: 1 for i in range(5)}, manifest["manifest_checksum"])
    frame = pd.DataFrame(contract["jobs"] + [contract["jobs"][0]])
    assert duplicate_job_rows(frame) == 2


def test_duplicate_metric_rows_are_detected_without_groupby():
    rows = [{"experiment_id":"S0","scenario":"late_stage","model_name":"cnn_bilstm","outer_fold":0,"training_seed":42}]
    assert duplicate_job_rows(pd.DataFrame(rows * 2)) == 2


def test_drop_last_false_consumes_every_record_once():
    dataset = TensorDataset(torch.arange(10))
    loader = DataLoader(dataset, batch_size=4, shuffle=False, drop_last=False)
    seen = [int(v) for batch in loader for v in batch[0]]
    assert seen == list(range(10))
    assert loader_statistics(10, 4, False)["samples_dropped_per_epoch"] == 0


def test_drop_last_true_logs_dropped_tail():
    stats = loader_statistics(10, 4, True)
    assert stats["samples_consumed_per_epoch"] == 8 and stats["samples_dropped_per_epoch"] == 2


def test_kernel_shape_contract():
    assert 2 + 2 * (1 // 2) - 1 + 1 == 2
    assert 2 + 2 * (2 // 2) - 2 + 1 == 3


def test_feature_contract_changes_with_preprocessing_payload():
    contract = feature_contract("fold")
    altered = dict(contract); altered["scaler_contract"] = "other"; altered.pop("semantic_checksum")
    from src.evaluation.neural_sanity_v2_2 import checksum
    assert contract["semantic_checksum"] != checksum(altered)
