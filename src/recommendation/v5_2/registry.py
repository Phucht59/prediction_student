from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from src.studies.v5_1.common.artifacts import atomic_write_json
from src.studies.v5_1.common.protocol import ROOT, sha256_file


def _uci_entry(dataset: str) -> dict[str, Any]:
    normalized = dataset.replace("-", "_")
    metrics_path = ROOT / "artifacts" / "v5_2" / normalized / "final_metrics.json"
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    metadata_path = ROOT / "artifacts/v5_2/uci/final_checkpoint_metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    comparator = {
        "student-mat": ("decision_tree", 0.9066544140712939),
        "student-por": ("random_forest", 0.8692436817866236),
    }[dataset]
    deep_id = f"v5.2-shared-three-head-{dataset}"
    ml_id = f"v5.1-{comparator[0].replace('_', '-')}-{dataset}"
    deep = {
        "registry_id": deep_id,
        "role": "recommendation_source_model",
        "candidate": "shared_trunk_subject_specific_three_heads_v5_2",
        "evidence_version": "v5.2",
        "macro_f1": metrics["metrics"]["classification"]["macro_f1"],
        "oof_path": f"artifacts/v5_2/{normalized}/oof_predictions.parquet",
        "oof_sha256": sha256_file(
            ROOT / f"artifacts/v5_2/{normalized}/oof_predictions.parquet"
        ),
        "checkpoint_paths": [row["path"] for row in metadata],
        "checkpoint_sha256": [row["sha256"] for row in metadata],
        "fixed_seeds": [42, 1201, 2026, 3407, 7319],
    }
    ml = {
        "registry_id": ml_id,
        "role": "recommendation_cross_check_model",
        "candidate": comparator[0],
        "evidence_version": "v5.1",
        "macro_f1": comparator[1],
        "oof_path": f"artifacts/v5_1/{normalized}/ml_oof_predictions.parquet",
        "oof_sha256": sha256_file(
            ROOT / f"artifacts/v5_1/{normalized}/ml_oof_predictions.parquet"
        ),
    }
    return {
        "dataset": dataset,
        "final_thesis_model": deep["candidate"],
        "final_operational_model": ml["candidate"],
        "recommendation_source_model": deep_id,
        "recommendation_cross_check_model": ml_id,
        "selection_metric": "macro_f1",
        "primary_metrics": metrics["metrics"]["classification"],
        "secondary_metrics": {
            "regression": metrics["metrics"]["regression"],
            "ordinal": metrics["metrics"]["ordinal"],
        },
        "stability_metrics": metrics["stability"],
        "efficiency_metrics": {
            "parameter_count_min": min(int(row["parameter_count"]) for row in metadata),
            "parameter_count_max": max(int(row["parameter_count"]) for row in metadata),
            "training_seconds_total": sum(float(row["runtime_seconds"]) for row in metadata),
            "peak_gpu_memory_bytes": max(int(row["peak_gpu_memory_bytes"]) for row in metadata),
        },
        "checkpoint_paths": deep["checkpoint_paths"],
        "checkpoint_sha256": deep["checkpoint_sha256"],
        "limitations": [
            "Deep did not exceed the strongest ML comparator on Macro-F1.",
            "Shared-learning contribution is not causally isolated from decoder/training changes.",
        ],
        "deep": deep,
        "ml": ml,
        "selection_note": "Deep remains the thesis recommendation source; strongest ML is an explicit cross-check.",
    }


def _oulad_entry() -> dict[str, Any]:
    v5_1_registry_path = ROOT / "artifacts/v5_1/oulad/model_registry.json"
    v5_1 = json.loads(v5_1_registry_path.read_text(encoding="utf-8"))
    deep_oof = ROOT / "artifacts/v5_1/oulad/oof_predictions.parquet"
    ml_oof = ROOT / "artifacts/oulad/v4/oulad-v4-f2-scientific-20260716-v1/oof_predictions.parquet"
    deep = {
        "registry_id": "v5.1-cnn-bilstm-full-oulad",
        "role": "recommendation_source_model",
        "candidate": "cnn_bilstm_full_ensemble",
        "evidence_version": "v5.1_retained_after_v5.2_gate_failure",
        "macro_f1": v5_1["macro_f1"],
        "oof_path": deep_oof.relative_to(ROOT).as_posix(),
        "oof_sha256": sha256_file(deep_oof),
        "checkpoint_paths": v5_1["checkpoint_paths"],
        "checkpoint_sha256": v5_1["checkpoint_sha256"],
        "fixed_seeds": v5_1["fixed_seeds"],
    }
    ml = {
        "registry_id": "v4-xgboost-ensemble-oulad",
        "role": "recommendation_cross_check_model",
        "candidate": "V4-XGB-ENS",
        "evidence_version": "v4_immutable_reused",
        "macro_f1": v5_1["strongest_ml_macro_f1"],
        "oof_path": ml_oof.relative_to(ROOT).as_posix(),
        "oof_sha256": sha256_file(ml_oof),
    }
    return {
        "dataset": "oulad",
        "final_thesis_model": deep["candidate"],
        "final_operational_model": ml["candidate"],
        "recommendation_source_model": deep["registry_id"],
        "recommendation_cross_check_model": ml["registry_id"],
        "selection_metric": "macro_f1",
        "primary_metrics": {"macro_f1": v5_1["macro_f1"]},
        "secondary_metrics": {"v5_2_outer_evaluation": "NOT_RUN_GATE_FAILED"},
        "stability_metrics": {"fixed_seeds": v5_1["fixed_seeds"]},
        "efficiency_metrics": {"parameter_counts": v5_1["parameter_counts"]},
        "checkpoint_paths": deep["checkpoint_paths"],
        "checkpoint_sha256": deep["checkpoint_sha256"],
        "limitations": [
            "V5.2 parallel candidates failed the preregistered inner gate.",
            "No V5.2 OULAD outer-test or future benchmark was run.",
        ],
        "deep": deep,
        "ml": ml,
        "selection_note": "V5.2 parallel candidates failed the preregistered inner gate; immutable V5.1 Deep evidence is retained.",
    }


def build_model_registry(output: Path | None = None) -> dict[str, Any]:
    output = output or ROOT / "artifacts/v5_2/final/model_registry.json"
    registry = {
        "schema_version": "v5_2_prediction_recommendation_registry_v1",
        "status": "COMPLETE",
        "prediction_source_policy": "cnn_bilstm_deep_primary_ml_cross_check_explicit",
        "datasets": {
            "student-mat": _uci_entry("student-mat"),
            "student-por": _uci_entry("student-por"),
            "oulad": _oulad_entry(),
        },
        "future_oulad": "LOCKED_NOT_EXECUTED",
        "hard_coded_legacy_candidate": False,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(output, registry)
    return registry


def lookup_registry(registry: dict[str, Any], dataset: str) -> tuple[dict[str, Any], dict[str, Any]]:
    try:
        entry = registry["datasets"][dataset]
        deep, ml = entry["deep"], entry["ml"]
    except KeyError as error:
        raise KeyError(f"Missing recommendation registry entry: {dataset}") from error
    if deep["role"] != "recommendation_source_model" or ml["role"] != "recommendation_cross_check_model":
        raise ValueError("Recommendation registry roles are invalid")
    return deep, ml


__all__ = ["build_model_registry", "lookup_registry"]
