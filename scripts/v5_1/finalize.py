from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.studies.v5_1.common.artifacts import (  # noqa: E402
    atomic_write_json,
    build_checksum_manifest,
)
from src.studies.v5_1.common.statistics import (  # noqa: E402
    paired_group_bootstrap,
    practical_verdict,
)

SEEDS = {42, 1201, 2026, 3407, 7319}
TARGETS = {"student_mat": 0.89, "student_por": 0.86, "oulad": 0.832}
V5_SCORES = {"student_mat": 0.8799168721, "student_por": 0.8491516177, "oulad": 0.8280026389}
STRONGEST_ML = {
    "student_mat": ("decision_tree_ensemble", 0.9066544140712939),
    "student_por": ("random_forest_ensemble", 0.8692436817866236),
    "oulad": ("V4-XGB-ENS", 0.8283814220319712),
}
PRIMARY = {
    "student_mat": ("cnn_bilstm_v5_1_transfer_selected", "decision_tree"),
    "student_por": ("cnn_bilstm_v5_1", "random_forest"),
}


def _uci_ensemble(
    path: Path, candidate: str, expected_seeds: set[int] | None = None
) -> pd.DataFrame:
    frame = pd.read_parquet(path)
    frame = frame.loc[frame.candidate == candidate].copy()
    registered = SEEDS if expected_seeds is None else expected_seeds
    if set(frame.seed.unique()) != registered:
        raise RuntimeError(f"Incomplete fixed-seed coverage for {candidate}")
    keys = ["record_id", "source_row", "outer_fold", "target"]
    probabilities = ["p_low", "p_medium", "p_high"]
    result = frame.groupby(keys, as_index=False)[probabilities].mean()
    result["prediction"] = result[probabilities].to_numpy().argmax(axis=1)
    if result.record_id.duplicated().any():
        raise RuntimeError(f"Duplicate UCI ensemble records for {candidate}")
    return result.sort_values("record_id").reset_index(drop=True)


def _aligned_bootstrap(
    left: pd.DataFrame,
    right: pd.DataFrame,
    *,
    dataset: str,
    left_name: str,
    right_name: str,
    group_column: str,
) -> dict[str, object]:
    columns = ["record_id", "target", "prediction", group_column]
    merged = left[columns].merge(
        right[columns], on="record_id", suffixes=("_left", "_right"), validate="one_to_one"
    )
    if len(merged) != len(left) or len(merged) != len(right):
        raise RuntimeError(f"Unaligned paired evidence: {dataset} {left_name} vs {right_name}")
    if not np.array_equal(merged.target_left, merged.target_right):
        raise RuntimeError(f"Target mismatch: {dataset} {left_name} vs {right_name}")
    comparison = paired_group_bootstrap(
        merged.target_left.to_numpy(),
        merged.prediction_left.to_numpy(),
        merged.prediction_right.to_numpy(),
        merged[f"{group_column}_left"].to_numpy(),
        replicates=5000,
        seed=3407,
    )
    return {
        "dataset": dataset,
        "left": left_name,
        "right": right_name,
        **comparison,
        "verdict": practical_verdict(comparison),
    }


def _uci_comparisons(dataset: str) -> list[dict[str, object]]:
    artifact = ROOT / "artifacts" / "v5_1" / dataset
    deep_name, ml_name = PRIMARY[dataset]
    deep = _uci_ensemble(artifact / "oof_predictions.parquet", deep_name)
    ml = _uci_ensemble(artifact / "ml_oof_predictions.parquet", ml_name)
    rows = [
        _aligned_bootstrap(
            deep,
            ml,
            dataset=dataset,
            left_name=f"{deep_name}_ensemble",
            right_name=f"{ml_name}_ensemble",
            group_column="source_row",
        )
    ]
    for ablation in ["cnn_only_v5_1", "bilstm_only_v5_1"]:
        ablation_frame = _uci_ensemble(
            artifact / "oof_predictions.parquet", ablation, {42, 2026, 3407}
        )
        rows.append(
            _aligned_bootstrap(
                deep,
                ablation_frame,
                dataset=dataset,
                left_name=f"{deep_name}_ensemble",
                right_name=f"{ablation}_ensemble",
                group_column="source_row",
            )
        )
    return rows


def _oulad_current(candidate: str) -> pd.DataFrame:
    frame = pd.read_parquet(ROOT / "artifacts/v5_1/oulad/oof_predictions.parquet")
    frame = frame.loc[frame.candidate == candidate].copy()
    result = frame[["record_id", "id_student", "target", "probability", "threshold"]].copy()
    result["prediction"] = (result.probability >= result.threshold).astype(int)
    return result.sort_values("record_id").reset_index(drop=True)


def _oulad_frozen(candidate: str, path: Path) -> pd.DataFrame:
    frame = pd.read_parquet(path)
    if "candidate_id" in frame:
        frame = frame.loc[frame.candidate_id == candidate]
        result = frame[["record_id", "id_student", "target_at_risk", "predicted_label"]].rename(
            columns={"target_at_risk": "target", "predicted_label": "prediction"}
        )
    else:
        frame = frame.loc[frame.candidate == candidate]
        result = frame[["record_id", "id_student", "target", "probability", "threshold"]].copy()
        result["prediction"] = (result.probability >= result.threshold).astype(int)
    return result.sort_values("record_id").reset_index(drop=True)


def _oulad_comparisons() -> list[dict[str, object]]:
    deep = _oulad_current("cnn_bilstm_full_ensemble")
    comparators = [
        (
            "V4-XGB-ENS",
            _oulad_frozen(
                "V4-XGB-ENS",
                ROOT / "artifacts/oulad/v4/oulad-v4-f2-scientific-20260716-v1/oof_predictions.parquet",
            ),
        ),
        (
            "v5_cnn_bilstm_ensemble",
            _oulad_frozen(
                "cnn_bilstm_ensemble", ROOT / "artifacts/v5/oulad/oof_predictions.parquet"
            ),
        ),
        ("cnn_only_ensemble", _oulad_current("cnn_only_ensemble")),
        ("bilstm_only_ensemble", _oulad_current("bilstm_only_ensemble")),
    ]
    return [
        _aligned_bootstrap(
            deep,
            right,
            dataset="oulad",
            left_name="cnn_bilstm_full_ensemble",
            right_name=name,
            group_column="id_student",
        )
        for name, right in comparators
    ]


def _headline(dataset: str) -> dict[str, object]:
    path = ROOT / "artifacts" / "v5_1" / dataset / "final_metrics.json"
    value = json.loads(path.read_text(encoding="utf-8"))
    if dataset == "oulad":
        row = next(item for item in value if item["candidate"] == "cnn_bilstm_full_ensemble")
        candidate = row["candidate"]
    else:
        row = value["metrics"]
        candidate = value["candidate"]
    score = float(row["macro_f1"])
    return {
        "candidate": candidate,
        "macro_f1": score,
        "directional_target": TARGETS[dataset],
        "directional_target_met": score >= TARGETS[dataset],
        "metrics": row,
    }


def _registry(dataset: str) -> dict[str, object]:
    artifact = ROOT / "artifacts" / "v5_1" / dataset
    checkpoints = json.loads((artifact / "checkpoint_metadata.json").read_text(encoding="utf-8"))
    headline = _headline(dataset)
    selected_base = str(headline["candidate"]).removesuffix("_ensemble")
    selected_checkpoints = [row for row in checkpoints if row["candidate"] == selected_base]
    strongest_name, strongest_score = STRONGEST_ML[dataset]
    limitations = [
        "Directional performance targets are not correctness gates.",
        "Future OULAD remains locked and no external generalization claim is allowed.",
    ]
    if dataset == "oulad":
        limitations.append("The V5.1 point estimate is below V5, XGBoost, and the 0.832 target.")
        limitations.append("The full hybrid is not distinguishable from BiLSTM-only by paired bootstrap.")
    else:
        limitations.append("The primary ML comparator has a higher point estimate with a confidence interval crossing zero.")
    registry = {
        "schema_version": "v5_1_model_registry_v1",
        "status": "COMPLETE",
        "dataset": dataset.replace("_", "-"),
        "selected_candidate": headline["candidate"],
        "final_thesis_model": headline["candidate"],
        "final_operational_model": strongest_name,
        "cnn_bilstm_v5_macro_f1": V5_SCORES[dataset],
        "cnn_bilstm_v5_1_macro_f1": headline["macro_f1"],
        "strongest_ml_model": strongest_name,
        "strongest_ml_macro_f1": strongest_score,
        "delta_vs_v5": headline["macro_f1"] - V5_SCORES[dataset],
        "delta_vs_ml": headline["macro_f1"] - strongest_score,
        "selected_reason": "Locked V5.1 research candidate; operational role remains with the stronger simple/ML comparator.",
        "limitations": limitations,
        "macro_f1": headline["macro_f1"],
        "directional_target": headline["directional_target"],
        "directional_target_met": headline["directional_target_met"],
        "fixed_seeds": sorted(SEEDS),
        "checkpoint_count": len(checkpoints),
        "selected_checkpoint_count": len(selected_checkpoints),
        "checkpoint_paths": [row["path"] for row in selected_checkpoints],
        "checkpoint_sha256": [row["sha256"] for row in selected_checkpoints],
        "parameter_counts": sorted({int(row["parameter_count"]) for row in checkpoints}),
        "max_replay_difference": max(float(row["replay_max_abs_difference"]) for row in checkpoints),
        "checkpoints": checkpoints,
        "future_accessed": False,
    }
    atomic_write_json(artifact / "model_registry.json", registry)
    atomic_write_json(artifact / "artifact_checksums.json", build_checksum_manifest(artifact))
    return registry


def main() -> int:
    report_root = ROOT / "reports/v5_1/final"
    artifact_root = ROOT / "artifacts/v5_1/final"
    report_root.mkdir(parents=True, exist_ok=True)
    artifact_root.mkdir(parents=True, exist_ok=True)

    comparisons = [
        *_uci_comparisons("student_mat"),
        *_uci_comparisons("student_por"),
        *_oulad_comparisons(),
    ]
    bootstrap = {
        "status": "COMPLETE",
        "replicates": 5000,
        "confidence_level": 0.95,
        "comparisons": comparisons,
        "future_accessed": False,
    }
    atomic_write_json(report_root / "paired_bootstrap.json", bootstrap)
    pd.DataFrame(comparisons).to_csv(report_root / "paired_bootstrap.csv", index=False)

    registries = {dataset: _registry(dataset) for dataset in ["student_mat", "student_por", "oulad"]}
    summary = {
        "status": "COMPLETE",
        "studies": {dataset: _headline(dataset) for dataset in registries},
        "model_registries": {
            dataset: f"artifacts/v5_1/{dataset}/model_registry.json" for dataset in registries
        },
        "paired_bootstrap": "reports/v5_1/final/paired_bootstrap.json",
        "future_oulad": "LOCKED_NOT_EXECUTED",
    }
    atomic_write_json(artifact_root / "model_registry.json", registries)
    atomic_write_json(artifact_root / "summary.json", summary)
    atomic_write_json(artifact_root / "artifact_checksums.json", build_checksum_manifest(artifact_root))
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
