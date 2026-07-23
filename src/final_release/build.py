"""Build canonical release tables exclusively from frozen final evidence."""

from __future__ import annotations

import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from src.evaluation.calibration import expected_calibration_error
from src.evaluation.classification import metrics_from_confusion
from src.evaluation.ranking import top_k_metrics
from src.final_release.catalog import (
    COMPARISON_MODELS,
    OFFICIAL_MODELS,
    RECOMMENDATION_SYSTEM,
)

ROOT = Path(__file__).resolve().parents[2]
FINAL_ROOT = ROOT / "artifacts" / "final"
REPORT_ROOT = ROOT / "reports" / "final"
SOURCE_COMMIT = "9611bfc1e7ff594f64f19fe3144a105216388954"
MISSING_REASON = "No frozen final prediction artifact"

UCI_METRICS = (
    "accuracy",
    "balanced_accuracy",
    "macro_precision",
    "macro_recall",
    "macro_f1",
    "weighted_f1",
    "pr_auc",
    "roc_auc",
    "brier",
    "nll",
    "ece",
)
OULAD_METRICS = UCI_METRICS + ("risk_precision", "risk_recall", "risk_f1")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def relative(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def sourced(
    value: Any, path: Path, calculation: str = "loaded_from_frozen_final_evidence"
) -> dict[str, Any]:
    return {
        "value": int(value)
        if isinstance(value, np.integer)
        else float(value)
        if isinstance(value, (float, np.floating))
        else value,
        "source_artifact": relative(path),
        "source_checksum": sha256(path),
        "calculation": calculation,
    }


def missing(reason: str = MISSING_REASON) -> dict[str, Any]:
    return {"value": None, "status": "N/A", "reason": reason}


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def prediction_extensions(path: Path, candidate: str, classes: int) -> dict[str, float]:
    frame = pd.read_parquet(path)
    frame = frame.loc[frame["candidate"] == candidate].copy()
    if frame.empty:
        return {}
    probability_columns = ["p_low", "p_medium", "p_high"]
    grouped = frame.groupby("record_id", sort=True)
    targets = grouped["target"].first().to_numpy(dtype=int)
    probabilities = grouped[probability_columns].mean().to_numpy(dtype=float)
    roc = roc_auc_score(targets, probabilities, multi_class="ovr", average="macro")
    one_hot = np.eye(classes, dtype=float)[targets]
    return {
        "roc_auc": float(roc),
        "brier": float(np.mean(np.sum((probabilities - one_hot) ** 2, axis=1))),
        "ece": expected_calibration_error(targets, probabilities),
    }


def metric_row(
    model_id: str,
    display_name: str,
    values: dict[str, Any] | None,
    source: Path | None,
    metric_names: tuple[str, ...],
    per_class: dict[str, Any] | None = None,
    confusion: list[list[int]] | None = None,
    extra: dict[str, tuple[float, Path]] | None = None,
) -> dict[str, Any]:
    metrics = {name: missing() for name in metric_names}
    if values is not None and source is not None:
        aliases = {
            "pr_auc": "macro_pr_auc",
            "risk_precision": "at_risk_precision",
            "risk_recall": "at_risk_recall",
            "risk_f1": "at_risk_f1",
        }
        for name in metric_names:
            source_name = aliases.get(name, name)
            raw_value = values.get(source_name)
            if raw_value is None and name == "pr_auc":
                raw_value = values.get("pr_auc")
            if raw_value is not None:
                metrics[name] = sourced(raw_value, source)
    if confusion is not None and source is not None:
        calculated = metrics_from_confusion(confusion)
        for name, value in calculated.items():
            if name in metrics:
                metrics[name] = sourced(
                    value, source, "recomputed_from_frozen_confusion_matrix"
                )
    for name, pair in (extra or {}).items():
        metrics[name] = sourced(pair[0], pair[1], "recomputed_from_frozen_predictions")
    return {
        "model_id": model_id,
        "model": display_name,
        "result_scope": "FINAL_PROBABILITY_ENSEMBLE",
        "metrics": metrics,
        "per_class": per_class or [],
        "confusion_matrix": (
            {
                "value": confusion,
                "source_artifact": relative(source),
                "source_checksum": sha256(source),
            }
            if confusion is not None and source is not None
            else missing()
        ),
        "top_k": [],
    }


def class_rows(
    values: dict[str, Any] | None, macro_f1: Any, labels: list[str], source: Path | None
) -> list[dict[str, Any]]:
    rows = []
    for label in labels:
        item = (values or {}).get(label.lower().replace("-", "_"))
        if item is None or source is None:
            rows.append(
                {
                    "class": label,
                    "precision": missing(),
                    "recall": missing(),
                    "f1": missing(),
                    "support": missing(),
                    "macro_f1": missing(),
                }
            )
        else:
            rows.append(
                {
                    "class": label,
                    "precision": sourced(item["precision"], source),
                    "recall": sourced(item["recall"], source),
                    "f1": sourced(item["f1"], source),
                    "support": sourced(item["support"], source),
                    "macro_f1": sourced(macro_f1, source),
                }
            )
    return rows


def binary_class_rows(
    confusion: list[list[int]] | None, macro_f1: Any, source: Path | None
) -> list[dict[str, Any]]:
    labels = ["Not-at-risk", "At-risk"]
    if confusion is None or source is None:
        return class_rows(None, None, labels, None)
    cm = np.asarray(confusion, dtype=float)
    rows = []
    for index, label in enumerate(labels):
        precision = cm[index, index] / cm[:, index].sum()
        recall = cm[index, index] / cm[index].sum()
        f1 = 2 * precision * recall / (precision + recall)
        rows.append(
            {
                "class": label,
                "precision": sourced(
                    float(precision), source, "recomputed_from_frozen_confusion_matrix"
                ),
                "recall": sourced(
                    float(recall), source, "recomputed_from_frozen_confusion_matrix"
                ),
                "f1": sourced(
                    float(f1), source, "recomputed_from_frozen_confusion_matrix"
                ),
                "support": sourced(
                    int(cm[index].sum()),
                    source,
                    "recomputed_from_frozen_confusion_matrix",
                ),
                "macro_f1": sourced(macro_f1, source),
            }
        )
    return rows


def uci_dataset(dataset: str) -> dict[str, Any]:
    root = ROOT / "artifacts" / "v5_1" / dataset
    deep_path = root / "final_metrics.json"
    ml_path = root / "ml_final_metrics.json"
    deep = load_json(deep_path)
    ml = load_json(ml_path)
    ablations = {entry["candidate"]: entry for entry in deep["ablation_metrics"]}
    ml_ensemble = {
        entry["candidate"].removesuffix("_ensemble"): entry
        for entry in ml
        if entry.get("seed") == -1
    }
    prediction_path = root / "oof_predictions.parquet"
    ml_prediction_path = root / "ml_oof_predictions.parquet"
    deep_candidate = (
        "cnn_bilstm_v5_1_transfer_selected"
        if dataset == "student_mat"
        else "cnn_bilstm_v5_1"
    )
    source_candidates = {
        "cnn_bilstm": (deep["metrics"], deep_path, deep_candidate, prediction_path),
        "cnn_only": (
            ablations.get("cnn_only_v5_1_ensemble"),
            deep_path,
            "cnn_only_v5_1",
            prediction_path,
        ),
        "bilstm_only": (
            ablations.get("bilstm_only_v5_1_ensemble"),
            deep_path,
            "bilstm_only_v5_1",
            prediction_path,
        ),
        "logistic_regression": (
            ml_ensemble.get("logistic_regression"),
            ml_path,
            "logistic_regression",
            ml_prediction_path,
        ),
        "decision_tree": (
            ml_ensemble.get("decision_tree"),
            ml_path,
            "decision_tree",
            ml_prediction_path,
        ),
        "random_forest": (
            ml_ensemble.get("random_forest"),
            ml_path,
            "random_forest",
            ml_prediction_path,
        ),
        "hist_gradient_boosting": (
            ml_ensemble.get("hist_gradient_boosting"),
            ml_path,
            "hist_gradient_boosting",
            ml_prediction_path,
        ),
        "svm": (ml_ensemble.get("svm"), ml_path, "svm", ml_prediction_path),
        "xgboost": (None, None, None, None),
    }
    rows = []
    for model_id, display_name in COMPARISON_MODELS:
        values, source, candidate, probability_source = source_candidates[model_id]
        extras: dict[str, tuple[float, Path]] = {}
        if (
            values is not None
            and probability_source is not None
            and candidate is not None
        ):
            extras = {
                name: (value, probability_source)
                for name, value in prediction_extensions(
                    probability_source, candidate, 3
                ).items()
            }
        confusion = values.get("confusion_matrix") if values else None
        per_class = class_rows(
            values.get("per_class") if values else None,
            values.get("macro_f1") if values else None,
            ["Low", "Medium", "High"],
            source,
        )
        rows.append(
            metric_row(
                model_id,
                display_name,
                values,
                source,
                UCI_METRICS,
                per_class,
                confusion,
                extras,
            )
        )
    return {
        "dataset": OFFICIAL_MODELS[dataset]["dataset"],
        "classes": ["Low", "Medium", "High"],
        "models": rows,
    }


def oulad_dataset() -> dict[str, Any]:
    official_path = ROOT / "artifacts" / "v6" / "prediction" / "final" / "metrics.json"
    official_rows = load_json(official_path)
    official = next(entry for entry in official_rows if entry.get("seed") == -1)
    ablation_path = ROOT / "artifacts" / "v5_1" / "oulad" / "final_metrics.json"
    ablations = load_json(ablation_path)
    ablation_map = {
        entry["candidate"]: entry for entry in ablations if entry.get("seed") == -1
    }
    comparator_path = ROOT / "artifacts" / "v5" / "oulad" / "final_metrics.csv"
    comparators = {
        row["candidate"]: row
        for row in csv.DictReader(comparator_path.open(encoding="utf-8"))
        if row.get("seed") == "-1"
    }
    for row in comparators.values():
        for key, value in list(row.items()):
            if key not in {
                "candidate",
                "threshold_scope",
                "source",
                "seed",
            } and value not in {"", None}:
                try:
                    row[key] = float(value)
                except ValueError:
                    pass
    sources = {
        "cnn_bilstm": (official, official_path),
        "cnn_only": (ablation_map.get("cnn_only_ensemble"), ablation_path),
        "bilstm_only": (ablation_map.get("bilstm_only_ensemble"), ablation_path),
        "logistic_regression": (
            comparators.get("logistic_regression"),
            comparator_path,
        ),
        "decision_tree": (None, None),
        "random_forest": (None, None),
        "hist_gradient_boosting": (
            comparators.get("hist_gradient_boosting"),
            comparator_path,
        ),
        "svm": (None, None),
        "xgboost": (comparators.get("xgboost"), comparator_path),
    }
    rows = []
    for model_id, display_name in COMPARISON_MODELS:
        values, source = sources[model_id]
        confusion = values.get("confusion_matrix") if values else None
        per_class = binary_class_rows(
            confusion, values.get("macro_f1") if values else None, source
        )
        row = metric_row(
            model_id, display_name, values, source, OULAD_METRICS, per_class, confusion
        )
        rows.append(row)

    probability_sources = {
        "cnn_bilstm": (
            ROOT
            / "artifacts"
            / "v6"
            / "prediction"
            / "final"
            / "seed_predictions.parquet",
            None,
        ),
        "cnn_only": (
            ROOT / "artifacts" / "v5_1" / "oulad" / "oof_predictions.parquet",
            "cnn_only",
        ),
        "bilstm_only": (
            ROOT / "artifacts" / "v5_1" / "oulad" / "oof_predictions.parquet",
            "bilstm_only",
        ),
    }
    for row in rows:
        if row["model_id"] not in probability_sources:
            row["top_k"] = [
                {
                    "budget": budget,
                    "precision": missing(),
                    "recall": missing(),
                    "f1": missing(),
                    "ndcg": missing(),
                }
                for budget in (0.05, 0.10, 0.20)
            ]
            continue
        path, candidate = probability_sources[row["model_id"]]
        frame = pd.read_parquet(path)
        if candidate is not None:
            frame = frame.loc[frame["candidate"] == candidate]
        grouped = (
            frame.groupby("record_id", sort=True)
            .agg(target=("target", "first"), probability=("probability", "mean"))
            .reset_index()
        )
        row["top_k"] = []
        for budget in (0.05, 0.10, 0.20):
            calculated = top_k_metrics(
                grouped.target.to_numpy(),
                grouped.probability.to_numpy(),
                grouped.record_id.to_numpy(),
                budget,
            )
            row["top_k"].append(
                {
                    "budget": budget,
                    **{
                        name: sourced(value, path, "recomputed_from_frozen_predictions")
                        for name, value in calculated.items()
                    },
                }
            )
    return {"dataset": "oulad", "classes": ["Not-at-risk", "At-risk"], "models": rows}


def recommendation_result() -> dict[str, Any]:
    technical_path = (
        ROOT / "artifacts" / "v6" / "recommendation" / "technical_metrics.json"
    )
    action_path = ROOT / "artifacts" / "v6" / "recommendation" / "action_metrics.json"
    technical = load_json(technical_path)
    action = load_json(action_path)
    return {
        **RECOMMENDATION_SYSTEM,
        "metrics": {
            key: sourced(technical[key], technical_path)
            for key in (
                "plans_generated",
                "coverage",
                "escalation_rate",
                "conflicts",
                "duplicate_plans",
                "workload_violations",
                "missing_lineage",
                "deterministic_replay",
            )
        },
        "expert_status": sourced(action["status"], action_path),
        "expert_metrics": {
            key: missing("Expert labels have not been supplied")
            for key in (
                "action_precision",
                "action_recall",
                "action_f1",
                "expert_agreement",
            )
        },
        "causal_effectiveness_claimed": sourced(False, technical_path),
    }


def build_payload() -> dict[str, Any]:
    completion_root = FINAL_ROOT / "comparator_completion"
    protocol_snapshot = load_json(completion_root / "protocol_snapshot.json")
    protocol_id = protocol_snapshot["protocol_id"]
    protocol_hash = protocol_snapshot["protocol_hash"]

    def completed_dataset(dataset: str) -> dict[str, Any]:
        root = completion_root / dataset
        evidence = load_json(root / "metrics.json")
        per_class_frame = pd.read_csv(root / "per_class.csv")
        confusion = load_json(root / "confusion_matrices.json")
        top_k_frame = (
            pd.read_csv(root / "top_k.csv")
            if (root / "top_k.csv").is_file()
            else pd.DataFrame()
        )
        rows = []
        for item in evidence["models"]:
            model_id = item["model_id"]
            metric_provenance = item["metric_provenance"]
            metrics = {
                name: {
                    "value": value,
                    **metric_provenance,
                }
                for name, value in item["metrics"].items()
            }
            class_rows = []
            selected_class = per_class_frame.loc[
                per_class_frame["model_id"] == model_id
            ]
            for _, class_item in selected_class.iterrows():
                class_rows.append(
                    {
                        "class": class_item["class"],
                        **{
                            name: {
                                "value": (
                                    int(class_item[name])
                                    if name == "support"
                                    else float(class_item[name])
                                ),
                                **metric_provenance,
                            }
                            for name in ("precision", "recall", "f1", "support")
                        },
                    }
                )
            top_k = []
            if not top_k_frame.empty:
                selected_top_k = top_k_frame.loc[
                    top_k_frame["model_id"] == model_id
                ]
                for _, top_item in selected_top_k.iterrows():
                    top_k.append(
                        {
                            "budget": float(top_item["budget"]),
                            "k": int(top_item["k"]),
                            "precision": {
                                "value": float(top_item["precision_at_k"]),
                                **metric_provenance,
                            },
                            "recall": {
                                "value": float(top_item["recall_at_k"]),
                                **metric_provenance,
                            },
                            "f1": {
                                "value": float(top_item["f1_at_k"]),
                                **metric_provenance,
                            },
                            "ndcg": {
                                "value": float(top_item["ndcg_at_k"]),
                                **metric_provenance,
                            },
                        }
                    )
            rows.append(
                {
                    "model_id": model_id,
                    "model": item["model"],
                    "result_scope": "FINAL_PROBABILITY_ENSEMBLE",
                    "metrics": metrics,
                    "per_class": class_rows,
                    "confusion_matrix": {
                        "value": confusion[model_id]["matrix"],
                        **metric_provenance,
                    },
                    "top_k": top_k,
                    "seed_stability": item["seed_stability"],
                    "evidence_origin": item["evidence_origin"],
                    "protocol_id": item["protocol_id"],
                    "source_artifacts": item["source_artifacts"],
                    "source_checksums": item["source_checksums"],
                }
            )
        expected = [model_id for model_id, _ in COMPARISON_MODELS]
        if [row["model_id"] for row in rows] != expected:
            raise RuntimeError(f"{dataset} completed model order mismatch")
        return {
            "dataset": "student-mat"
            if dataset == "student_mat"
            else "student-por"
            if dataset == "student_por"
            else "oulad",
            "classes": ["Low", "Medium", "High"]
            if dataset != "oulad"
            else ["Not-at-risk", "At-risk"],
            "models": rows,
        }

    return {
        "schema_version": "final_results_v2",
        "generated_from_validated_evidence": True,
        "comparator_completion_performed": True,
        "official_deep_models_retrained": False,
        "future_oulad_executed": False,
        "training_performed": True,
        "missing_metric_policy": "FAIL_ON_APPLICABLE_NA",
        "completion_protocol_id": protocol_id,
        "completion_protocol_hash": protocol_hash,
        "comparison_models": [model_id for model_id, _ in COMPARISON_MODELS],
        "official_models": {
            key: value["model_id"] for key, value in OFFICIAL_MODELS.items()
        },
        "datasets": {
            "student_mat": completed_dataset("student_mat"),
            "student_por": completed_dataset("student_por"),
            "oulad": completed_dataset("oulad"),
        },
        "recommendation": recommendation_result(),
        "claim_boundaries": [
            "Final tables report only final outer OOF or final probability ensembles.",
            "No causal recommendation effectiveness claim is made without expert or intervention labels.",
            "Future OULAD remains LOCKED_NOT_EXECUTED.",
            "Cross-domain advantage is not claimed where frozen evidence did not establish it.",
        ],
        "future_oulad": "LOCKED_NOT_EXECUTED",
        "source_commit": SOURCE_COMMIT,
    }


def write_results(payload: dict[str, Any]) -> None:
    FINAL_ROOT.mkdir(parents=True, exist_ok=True)
    (FINAL_ROOT / "final_results.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    metric_names = sorted(
        {
            name
            for dataset in payload["datasets"].values()
            for row in dataset["models"]
            for name in row["metrics"]
        }
    )
    columns = ["dataset", "model_id", "model", "result_scope"] + metric_names
    with (FINAL_ROOT / "final_results.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for dataset_id, dataset in payload["datasets"].items():
            for row in dataset["models"]:
                writer.writerow(
                    {
                        "dataset": dataset_id,
                        "model_id": row["model_id"],
                        "model": row["model"],
                        "result_scope": row["result_scope"],
                        **{
                            name: row["metrics"].get(name, missing())["value"]
                            for name in metric_names
                        },
                    }
                )


def build_registry() -> dict[str, Any]:
    evidence = {
        "student_mat": ROOT
        / "artifacts"
        / "v5_1"
        / "student_mat"
        / "final_metrics.json",
        "student_por": ROOT
        / "artifacts"
        / "v5_1"
        / "student_por"
        / "final_metrics.json",
        "oulad": ROOT / "artifacts" / "v6" / "prediction" / "final" / "metrics.json",
    }
    registry: dict[str, Any] = {"schema_version": "official_model_registry_v1"}
    for dataset, metadata in OFFICIAL_MODELS.items():
        source = evidence[dataset]
        checkpoint_manifest = source.parent / "checkpoint_metadata.json"
        if not checkpoint_manifest.is_file():
            checkpoint_manifest = (
                ROOT / "artifacts" / "v6" / "prediction" / "checkpoint_registry.json"
            )
        registry[dataset] = {
            **metadata,
            "status": "selected",
            "source_commit": SOURCE_COMMIT,
            "source_artifact_path": relative(source),
            "source_checksum": sha256(source),
            "checkpoint_checksum": sha256(checkpoint_manifest)
            if checkpoint_manifest.is_file()
            else None,
            "feature_contract_checksum": sha256(
                ROOT / "configs" / "v6" / "integrated_system_protocol.yaml"
            )
            if dataset == "oulad"
            else sha256(ROOT / "configs" / "v5_1" / "project_v5_1_protocol.yaml"),
        }
    registry["recommendation"] = {
        **RECOMMENDATION_SYSTEM,
        "source_commit": SOURCE_COMMIT,
        "source_artifact_path": "artifacts/v6/recommendation/technical_metrics.json",
        "source_checksum": sha256(
            ROOT / "artifacts" / "v6" / "recommendation" / "technical_metrics.json"
        ),
    }
    (FINAL_ROOT / "model_registry.json").write_text(
        json.dumps(registry, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return registry


def main() -> int:
    payload = build_payload()
    write_results(payload)
    registry = build_registry()
    for dataset in (*OFFICIAL_MODELS, "recommendation"):
        target = FINAL_ROOT / dataset
        target.mkdir(parents=True, exist_ok=True)
        (target / "index.json").write_text(
            json.dumps(
                {
                    "canonical_results": "../final_results.json",
                    "registry": registry[dataset],
                },
                indent=2,
                ensure_ascii=False,
            )
            + "\n",
            encoding="utf-8",
        )
    print(
        json.dumps(
            {
                "status": "PASS",
                "training_performed": True,
                "comparator_completion_performed": True,
                "official_deep_models_retrained": False,
                "dataset_rows": {
                    key: len(value["models"])
                    for key, value in payload["datasets"].items()
                },
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
