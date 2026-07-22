import json
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[2]


def test_final_scope_and_class_averages() -> None:
    payload = json.loads(
        (ROOT / "artifacts/final/final_results.json").read_text(encoding="utf-8")
    )
    for dataset in payload["datasets"].values():
        for row in dataset["models"]:
            assert row["result_scope"] == "FINAL_PROBABILITY_ENSEMBLE"
            f1_values = [
                item["f1"]["value"]
                for item in row["per_class"]
                if item["f1"]["value"] is not None
            ]
            if f1_values:
                assert row["metrics"]["macro_f1"]["value"] == pytest.approx(
                    sum(f1_values) / len(f1_values)
                )
