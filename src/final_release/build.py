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


def write_text_lf(path: Path, content: str) -> None:
    """Write repository-canonical UTF-8 text on every operating system."""
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(content)


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


def recommendation_result() -> dict[str, Any]:
    technical_path = (
        FINAL_ROOT / "recommendation" / "recommendation_technical_validation.json"
    )
    action_path = (
        FINAL_ROOT
        / "recommendation"
        / "expert_evaluation"
        / "expert_metrics.json"
    )
    technical = load_json(technical_path)
    action = load_json(action_path)
    return {
        **RECOMMENDATION_SYSTEM,
        "metrics": {
            "records": sourced(technical["records"], technical_path),
            "generated": sourced(technical["status_counts"]["GENERATED"], technical_path),
            "partial_evidence": sourced(
                technical["status_counts"]["PARTIAL_EVIDENCE"], technical_path
            ),
            "abstained": sourced(
                technical["status_counts"]["ABSTAINED"], technical_path
            ),
            "generated_or_partial": sourced(
                technical["coverage_generated_or_partial"], technical_path
            ),
            "abstention": sourced(technical["abstention_rate"], technical_path),
            "workload_violations": sourced(
                technical["workload_violations"], technical_path
            ),
            "action_cap_violations": sourced(
                technical["action_cap_violations"], technical_path
            ),
            "duplicates": sourced(
                technical["duplicate_action_violations"], technical_path
            ),
            "missing_lineage": sourced(
                technical["missing_action_lineage"], technical_path
            ),
            "post_cutoff_usage": sourced(technical["post_cutoff_used"], technical_path),
            "sensitive_usage": sourced(
                technical["sensitive_attributes_in_payload_or_reasoning"],
                technical_path,
            ),
            "withdrawal_mechanism_usage": sourced(
                technical["withdrawal_action_paths"], technical_path
            ),
            "deterministic_replay": sourced(
                technical["deterministic_replay"], technical_path
            ),
        },
        "expert_status": sourced(
            "PENDING_EXPERT_LABELS",
            action_path,
            "normalized_from_pending_real_expert_labels",
        ),
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
        teacher_root = FINAL_ROOT / "teacher_feedback_validation"
        feature_contracts = load_json(completion_root / "feature_contract.json")[
            "contracts"
        ]
        split_hashes = load_json(
            completion_root / "split_manifest_checksums.json"
        )

        def teacher_row(
            raw: dict[str, Any],
            prediction_path: Path,
            seed_path: Path,
            *,
            evidence_origin: str,
        ) -> dict[str, Any]:
            model_id = raw["model_id"]
            labels = (
                ["Not-at-risk", "At-risk"]
                if dataset == "oulad"
                else ["Low", "Medium", "High"]
            )
            protocol_path = ROOT / "configs" / "final" / "mlp_comparator.yaml"
            if dataset != "oulad" and model_id != "mlp":
                protocol_path = (
                    ROOT / "configs" / "final" / "uci_timing_scenarios.yaml"
                )
            provenance = {
                "calculation_method": (
                    "recomputed_from_record_aligned_ensemble_probability"
                ),
                "feature_contract_hash": feature_contracts[dataset]["sha256"],
                "protocol_hash": sha256(protocol_path),
                "source_artifact": relative(prediction_path),
                "source_checksum": sha256(prediction_path),
                "split_manifest_hash": split_hashes[dataset],
            }
            metric_values = {
                name: {"value": raw[name], **provenance}
                for name in (
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
            }
            if dataset == "oulad":
                at_risk = raw["per_class"]["At-risk"]
                metric_values.update(
                    {
                        "risk_precision": {
                            "value": at_risk["precision"],
                            **provenance,
                        },
                        "risk_recall": {
                            "value": at_risk["recall"],
                            **provenance,
                        },
                        "risk_f1": {"value": at_risk["f1"], **provenance},
                    }
                )
            class_values = [
                {
                    "class": label,
                    **{
                        name: {
                            "value": raw["per_class"][label][name],
                            **provenance,
                        }
                        for name in ("precision", "recall", "f1", "support")
                    },
                }
                for label in labels
            ]
            top_k: list[dict[str, Any]] = []
            if dataset == "oulad":
                prediction = pd.read_parquet(prediction_path)
                for budget in (0.01, 0.05, 0.10):
                    values = top_k_metrics(
                        prediction["true_label"].to_numpy(dtype=int),
                        prediction["p_at_risk"].to_numpy(dtype=float),
                        prediction["record_id"].to_numpy(),
                        budget,
                    )
                    top_k.append(
                        {
                            "budget": budget,
                            "k": int(math.ceil(len(prediction) * budget)),
                            **{
                                key: {
                                    "value": float(values[source]),
                                    **provenance,
                                }
                                for key, source in (
                                    ("precision", "precision"),
                                    ("recall", "recall"),
                                    ("f1", "f1"),
                                    ("ndcg", "ndcg"),
                                )
                            },
                        }
                    )
            sources = [prediction_path, seed_path, protocol_path]
            return {
                "model_id": model_id,
                "model": raw["model"],
                "result_scope": "FINAL_PROBABILITY_ENSEMBLE",
                "metrics": metric_values,
                "per_class": class_values,
                "confusion_matrix": {
                    "value": raw["confusion_matrix"],
                    **provenance,
                },
                "top_k": top_k,
                "seed_stability": raw["seed_stability"],
                "evidence_origin": evidence_origin,
                "protocol_id": "teacher-feedback-completion-20260728-v1",
                "source_artifacts": [relative(path) for path in sources],
                "source_checksums": {
                    relative(path): sha256(path) for path in sources
                },
            }

        if dataset in {"student_mat", "student_por"}:
            safe_root = teacher_root / "safe_uci_comparators" / dataset
            safe = load_json(safe_root / "metrics.json")
            safe_rows = {
                item["model_id"]: teacher_row(
                    item,
                    safe_root / "oof_predictions.parquet",
                    safe_root / "seed_predictions.parquet",
                    evidence_origin="teacher_feedback_safe_revalidation",
                )
                for item in safe["models"]
            }
            rows = [
                safe_rows.get(row["model_id"], row)
                for row in rows
            ]
            rows.append(safe_rows["mlp"])
        else:
            mlp_root = teacher_root / "mlp_comparator" / "oulad"
            rows.append(
                teacher_row(
                    load_json(mlp_root / "metrics.json"),
                    mlp_root / "oof_predictions.parquet",
                    mlp_root / "seed_predictions.parquet",
                    evidence_origin="teacher_feedback_mlp_completion",
                )
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
        "training_performed": False,
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
    write_text_lf(
        FINAL_ROOT / "final_results.json",
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
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
        writer = csv.DictWriter(handle, fieldnames=columns, lineterminator="\n")
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
        "student_mat": FINAL_ROOT / "metrics" / "cnn_bilstm_mat.json",
        "student_por": FINAL_ROOT / "metrics" / "cnn_bilstm_por.json",
        "oulad": FINAL_ROOT / "metrics" / "cnn_bilstm_oulad.json",
    }
    registry: dict[str, Any] = {"schema_version": "official_model_registry_v1"}
    for dataset, metadata in OFFICIAL_MODELS.items():
        source = evidence[dataset]
        checkpoint_manifest = FINAL_ROOT / "checksums" / "checkpoint_manifest.json"
        registry[metadata["model_id"]] = {
            **metadata,
            "status": "selected",
            "source_commit": SOURCE_COMMIT,
            "source_artifact_path": relative(source),
            "source_checksum": sha256(source),
            "checkpoint_checksum": sha256(checkpoint_manifest)
            if checkpoint_manifest.is_file()
            else None,
            "feature_contract_checksum": sha256(
                ROOT
                / "configs"
                / "final"
                / f"{metadata['model_id']}.yaml"
            ),
        }
    registry["recommendation"] = {
        **RECOMMENDATION_SYSTEM,
        "source_commit": SOURCE_COMMIT,
        "source_artifact_path": "artifacts/final/recommendation/recommendation_technical_validation.json",
        "source_checksum": sha256(
            FINAL_ROOT
            / "recommendation"
            / "recommendation_technical_validation.json"
        ),
    }
    registry["comparator_catalog"] = {
        model_id: {
            "model_id": model_id,
            "display_name": display_name,
            "role": "official_selected_family"
            if model_id == "cnn_bilstm"
            else "standalone_comparator",
            "datasets": ["student_mat", "student_por", "oulad"],
            "selected": model_id == "cnn_bilstm",
        }
        for model_id, display_name in COMPARISON_MODELS
    }
    write_text_lf(
        FINAL_ROOT / "model_registry.json",
        json.dumps(registry, indent=2, ensure_ascii=False) + "\n",
    )
    return registry


def main() -> int:
    payload = build_payload()
    write_results(payload)
    registry = build_registry()
    for dataset in (*OFFICIAL_MODELS, "recommendation"):
        target = FINAL_ROOT / dataset
        target.mkdir(parents=True, exist_ok=True)
        write_text_lf(
            target / "index.json",
            json.dumps(
                {
                    "canonical_results": "../final_results.json",
                    "registry": registry[
                        OFFICIAL_MODELS[dataset]["model_id"]
                        if dataset in OFFICIAL_MODELS
                        else dataset
                    ],
                },
                indent=2,
                ensure_ascii=False,
            )
            + "\n",
        )
    print(
        json.dumps(
            {
                "status": "PASS",
                "training_performed": False,
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
