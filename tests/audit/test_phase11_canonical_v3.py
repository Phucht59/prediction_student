from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.canonical_v3.benchmark import PRIMARY_MODELS, validate_preflight
from src.canonical_v3.metrics import binary_metrics, multiclass_metrics
from src.canonical_v3.oulad_data import CANONICAL_STAGES, build_canonical_bundle
from src.pipelines import oulad

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "artifacts" / "canonical_v3"


def test_preflight_replays_frozen_hashes() -> None:
    result = validate_preflight()
    assert result["status"] == "PASS"
    assert result["architecture_search"] is False
    assert result["outer_labels_used_for_selection"] is False


def test_canonical_bundle_is_monotonic_and_score_free() -> None:
    bundle = build_canonical_bundle()
    assert tuple(bundle.stages) == CANONICAL_STAGES
    cutoff = bundle.cutoff.loc[:, list(CANONICAL_STAGES)].to_numpy()
    assert np.all(cutoff[:, :-1] <= cutoff[:, 1:])
    score_columns = [
        oulad.CHANNELS.index("available_score_count"),
        oulad.CHANNELS.index("cumulative_mean_score"),
        oulad.CHANNELS.index("cumulative_weighted_score"),
    ]
    for stage in CANONICAL_STAGES:
        data = bundle.stages[stage]
        assert np.all(data.sequence[:, :, score_columns] == 0)
        missing = oulad.CHANNELS.index("score_missing_mask")
        assert np.all(data.sequence[:, :, missing][data.mask] == 1)


def test_final_is_superset_cutoff_and_has_no_75_only_feature() -> None:
    audit = json.loads((OUT / "oulad_feature_monotonicity.json").read_text(encoding="utf-8"))
    assert audit["75_only_features"] == []
    assert audit["relations"]["L1_LATE_75PCT_subset_FINAL"] is True
    assert audit["old_comparison"]["same_score_policy"] is True


def test_metric_contract_is_complete() -> None:
    common = {
        "accuracy",
        "balanced_accuracy",
        "macro_precision",
        "macro_recall",
        "macro_f1",
        "weighted_precision",
        "weighted_recall",
        "weighted_f1",
        "pr_auc",
        "roc_auc",
        "nll",
        "brier",
        "ece",
    }
    multiclass = multiclass_metrics(
        np.array([0, 1, 2, 0, 1, 2]),
        np.array(
            [
                [0.8, 0.1, 0.1],
                [0.1, 0.8, 0.1],
                [0.1, 0.1, 0.8],
                [0.7, 0.2, 0.1],
                [0.2, 0.7, 0.1],
                [0.1, 0.2, 0.7],
            ]
        ),
    )
    binary = binary_metrics(
        np.array([0, 1, 0, 1]), np.array([0.1, 0.9, 0.2, 0.8]), 0.5
    )
    assert common.issubset(multiclass)
    assert common.issubset(binary)
    assert len(multiclass["per_class"]) == 3
    assert {"risk_precision", "risk_recall", "risk_f1", "specificity"}.issubset(binary)


def test_old_replay_fold_membership_is_identical_across_models() -> None:
    uci = pd.read_parquet(
        ROOT / "artifacts/final/unified_stage_aware_uci/predictions.parquet"
    )
    for dataset in ("student_mat", "student_por"):
        current = uci.loc[uci.dataset.eq(dataset)]
        hashes = []
        for _, model in current.groupby("model_family"):
            pairs = model[["record_id", "outer_fold"]].drop_duplicates().sort_values("record_id")
            hashes.append(hash(tuple(map(tuple, pairs.to_numpy()))))
        assert len(set(hashes)) == 1
    oulad_frame = pd.read_parquet(
        ROOT / "artifacts/final/unified_stage_aware_oulad/predictions.parquet"
    )
    for stage, current in oulad_frame.groupby("prediction_stage"):
        hashes = []
        for _, model in current.groupby("model_family"):
            pairs = model[["base_record_id", "outer_fold"]].drop_duplicates().sort_values(
                "base_record_id"
            )
            hashes.append(hash(tuple(map(tuple, pairs.to_numpy()))))
        assert len(set(hashes)) == 1, stage


def test_primary_model_count_and_architecture_freeze() -> None:
    freeze = json.loads((OUT / "CANONICAL_BENCHMARK_FREEZE.json").read_text(encoding="utf-8"))
    assert len(PRIMARY_MODELS) == 8
    assert freeze["oulad_architecture_hash"] == (
        "df5cd885b96e5cea4b840bfc5ca59c08c095f5887df8dd8dcef738edfe8bf70e"
    )
    assert freeze["oulad_parameter_count"] == 160492
    assert freeze["architecture_search"] is False
