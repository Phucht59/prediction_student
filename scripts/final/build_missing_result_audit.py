#!/usr/bin/env python
"""Expand the pre-training final_results_v1 gaps into a cell-level audit."""

from __future__ import annotations

import json
import os
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "artifacts/final/final_results.json"
TARGET = ROOT / "artifacts/final/comparator_completion/missing_result_audit.json"


def action(dataset: str, model: str) -> tuple[str, str | None]:
    if dataset in {"student_mat", "student_por"} and model == "xgboost":
        return "TRAIN_COMPLETION_MODEL", None
    if dataset == "oulad" and model in {
        "logistic_regression",
        "hist_gradient_boosting",
        "xgboost",
    }:
        return "TRAIN_COMPLETION_MODEL", "DO_NOT_IMPORT"
    if dataset == "oulad" and model in {
        "decision_tree",
        "random_forest",
        "svm",
    }:
        return "TRAIN_COMPLETION_MODEL", None
    return "DERIVE_ONLY", None


def main() -> None:
    source = json.loads(SOURCE.read_text(encoding="utf-8"))
    audit = json.loads(TARGET.read_text(encoding="utf-8"))
    cells = []
    for dataset, dataset_value in source["datasets"].items():
        for model in dataset_value["models"]:
            primary, historical = action(dataset, model["model_id"])
            for metric, value in model["metrics"].items():
                if value.get("value") is None:
                    cells.append(
                        {
                            "dataset": dataset,
                            "model_id": model["model_id"],
                            "table": "overall",
                            "cell": metric,
                            "action": primary,
                            "historical_artifact_action": historical,
                        }
                    )
            for class_index, class_row in enumerate(model.get("per_class", [])):
                for metric, value in class_row.items():
                    if (
                        metric != "class"
                        and isinstance(value, dict)
                        and value.get("value") is None
                    ):
                        cells.append(
                            {
                                "dataset": dataset,
                                "model_id": model["model_id"],
                                "table": "per_class",
                                "class_index": class_index,
                                "cell": metric,
                                "action": primary,
                                "historical_artifact_action": historical,
                            }
                        )
            confusion = model.get("confusion_matrix", {})
            if isinstance(confusion, dict) and confusion.get("value") is None:
                cells.append(
                    {
                        "dataset": dataset,
                        "model_id": model["model_id"],
                        "table": "confusion_matrix",
                        "cell": "matrix",
                        "action": primary,
                        "historical_artifact_action": historical,
                    }
                )
            for budget_index, top_row in enumerate(model.get("top_k", [])):
                for metric, value in top_row.items():
                    if isinstance(value, dict) and value.get("value") is None:
                        cells.append(
                            {
                                "dataset": dataset,
                                "model_id": model["model_id"],
                                "table": "top_k",
                                "budget_index": budget_index,
                                "cell": metric,
                                "action": primary,
                                "historical_artifact_action": historical,
                            }
                        )
    if len(cells) != audit["serialized_missing_field_count"]["total"]:
        raise RuntimeError(
            f"Cell inventory mismatch: {len(cells)} != "
            f"{audit['serialized_missing_field_count']['total']}"
        )
    audit["missing_cells"] = cells
    audit["all_missing_cells_classified"] = True
    temporary = TARGET.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(audit, indent=2), encoding="utf-8")
    os.replace(temporary, TARGET)


if __name__ == "__main__":
    main()
