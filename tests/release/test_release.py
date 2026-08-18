from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path

import pytest

from src.release.build import ROOT, build_payload
from src.release.catalog import COMPARISON_MODELS, OFFICIAL_MODELS


@pytest.fixture(scope="module")
def payload() -> dict:
    return json.loads(
        (ROOT / "artifacts/final/final_results.json").read_text(encoding="utf-8")
    )


def test_official_cnn_bilstm_names() -> None:
    assert [item["official_name"] for item in OFFICIAL_MODELS.values()] == [
        "CNN-BiLSTM MAT",
        "CNN-BiLSTM POR",
        "CNN-BiLSTM OULAD",
    ]


def test_same_model_catalog_for_all_datasets(payload: dict) -> None:
    expected = [item[0] for item in COMPARISON_MODELS]
    assert all(
        [row["model_id"] for row in dataset["models"]] == expected
        for dataset in payload["datasets"].values()
    )


def test_decision_tree_present_in_oulad_table(payload: dict) -> None:
    assert "decision_tree" in [
        row["model_id"] for row in payload["datasets"]["oulad"]["models"]
    ]


def test_random_forest_present_in_oulad_table(payload: dict) -> None:
    assert "random_forest" in [
        row["model_id"] for row in payload["datasets"]["oulad"]["models"]
    ]


def test_precision_present_in_overall_tables() -> None:
    for name in (
        "STUDENT_MAT_RESULTS.md",
        "STUDENT_POR_RESULTS.md",
        "OULAD_RESULTS.md",
    ):
        assert "| Precision |" in (ROOT / "reports/final" / name).read_text(
            encoding="utf-8"
        )


def test_macro_f1_absent_from_per_class_tables() -> None:
    for name in (
        "STUDENT_MAT_RESULTS.md",
        "STUDENT_POR_RESULTS.md",
        "OULAD_RESULTS.md",
    ):
        assert "Model Macro-F1" not in (ROOT / "reports/final" / name).read_text(
            encoding="utf-8"
        )


def test_no_lab_version_in_readme() -> None:
    assert not re.search(
        r"\bV(?:4|5|6)(?:[._-]\d+)?\b",
        (ROOT / "README.md").read_text(encoding="utf-8"),
        re.I,
    )


def test_no_lab_version_in_project() -> None:
    assert not re.search(
        r"\bV(?:4|5|6)(?:[._-]\d+)?\b",
        (ROOT / "PROJECT.md").read_text(encoding="utf-8"),
        re.I,
    )


def test_test_lab_is_gitignored() -> None:
    assert (
        subprocess.run(
            ["git", "check-ignore", "test_lab/"], cwd=ROOT, capture_output=True
        ).returncode
        == 0
    )


def test_test_lab_has_no_tracked_files() -> None:
    assert not subprocess.check_output(
        ["git", "ls-files", "test_lab"], cwd=ROOT, text=True
    ).strip()


def test_final_metrics_have_sources(payload: dict) -> None:
    for dataset in payload["datasets"].values():
        for row in dataset["models"]:
            for metric in row["metrics"].values():
                if metric["value"] is not None:
                    assert metric.get("source_artifact") and metric.get(
                        "source_checksum"
                    )


def test_no_applicable_missing_model_metrics(payload: dict) -> None:
    missing = [
        metric
        for dataset in payload["datasets"].values()
        for row in dataset["models"]
        for metric in row["metrics"].values()
        if metric.get("status") == "N/A"
    ]
    assert not missing


def test_no_fabricated_metrics(payload: dict) -> None:
    assert payload == build_payload()


def test_per_class_matches_predictions(payload: dict) -> None:
    for dataset in payload["datasets"].values():
        for row in dataset["models"]:
            matrix = row["confusion_matrix"].get("value")
            if matrix is not None:
                assert sum(
                    item["support"]["value"] for item in row["per_class"]
                ) == sum(map(sum, matrix))


def test_macro_f1_matches_per_class(payload: dict) -> None:
    for dataset in payload["datasets"].values():
        for row in dataset["models"]:
            values = [
                item["f1"]["value"]
                for item in row["per_class"]
                if item["f1"]["value"] is not None
            ]
            if values:
                assert sum(values) / len(values) == pytest.approx(
                    row["metrics"]["macro_f1"]["value"]
                )


def test_top_k_requires_probability(payload: dict) -> None:
    for row in payload["datasets"]["oulad"]["models"]:
        for item in row["top_k"]:
            for metric in (item["precision"], item["recall"], item["f1"], item["ndcg"]):
                assert (
                    metric["value"] is None
                    or metric.get("calculation_method")
                    == "recomputed_from_record_aligned_ensemble_probability"
                )


def test_future_oulad_locked(payload: dict) -> None:
    assert payload["future_oulad"] == "LOCKED_NOT_EXECUTED"


def test_expert_status_not_fabricated(payload: dict) -> None:
    assert (
        payload["recommendation"]["expert_status"]["value"] == "PENDING_EXPERT_LABELS"
    )
