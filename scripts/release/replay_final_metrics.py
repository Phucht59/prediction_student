"""Build the final thesis release from frozen UCI and Canonical V3 predictions.

This entry point intentionally never trains, tunes, or selects a model.  It
only replays stored probabilities under the authorities in
``configs/final/final_model_authority.yaml``.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.evaluation.metrics import evaluate_multiclass

AUTHORITY_PATH = ROOT / "configs/final/final_model_authority.yaml"
RELEASE = ROOT / "artifacts/final_release"
REPORT = ROOT / "reports/final/thesis_v3"
PRIMARY = (
    "logistic_regression",
    "decision_tree",
    "random_forest",
    "hist_gradient_boosting",
    "svm",
    "xgboost",
    "mlp",
    "cnn_bilstm",
)
METRICS = (
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
)


def _json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, lineterminator="\n")


def _markdown_table(frame: pd.DataFrame, columns: list[str]) -> str:
    header = "| " + " | ".join(columns) + " |"
    divider = "|" + "|".join("---" for _ in columns) + "|"
    rows = []
    for _, row in frame.loc[:, columns].iterrows():
        values = []
        for column in columns:
            value = row[column]
            values.append(f"{value:.6f}" if isinstance(value, float) else str(value))
        rows.append("| " + " | ".join(values) + " |")
    return "\n".join([header, divider, *rows])


def _canonical_model_id(value: str) -> str:
    return value.replace("_mat", "").replace("_por", "")


def _mean_probability(frame: pd.DataFrame, model: str) -> pd.DataFrame:
    current = frame.loc[frame.model_id.eq(model)].copy()
    probability = ["p_low", "p_medium", "p_high"]
    grouped = current.groupby(["record_id", "outer_fold", "true_label"], as_index=False)[probability]
    return grouped.mean()


def _uci_main(dataset: str) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    completion = pd.read_parquet(
        ROOT / f"artifacts/final/comparator_completion/{dataset}/oof_predictions.parquet"
    )
    xgboost = pd.read_parquet(
        ROOT / f"artifacts/final/comparator_completion/{dataset}/xgboost_oof_predictions.parquet"
    )
    xgboost["model_id"] = xgboost.model_id.map(_canonical_model_id)
    mlp = pd.read_parquet(
        ROOT / f"artifacts/final/teacher_feedback_validation/mlp_comparator/{dataset}/oof_predictions.parquet"
    ).rename(columns={"target": "true_label"})
    metric_source = _json(ROOT / f"artifacts/final/comparator_completion/{dataset}/metrics.json")
    expected = {item["model_id"]: item["metrics"] for item in metric_source["models"]}
    expected["mlp"] = _json(
        ROOT / f"artifacts/final/teacher_feedback_validation/mlp_comparator/{dataset}/metrics.json"
    )
    sources = {
        "logistic_regression": completion,
        "decision_tree": completion,
        "random_forest": completion,
        "hist_gradient_boosting": completion,
        "svm": completion,
        "cnn_bilstm": completion,
        "xgboost": xgboost,
        "mlp": mlp,
    }
    rows: list[dict[str, Any]] = []
    class_rows: list[dict[str, Any]] = []
    matrices: dict[str, Any] = {}
    for model, source in sources.items():
        data = _mean_probability(source, model)
        result = evaluate_multiclass(
            data.true_label.to_numpy(), data[["p_low", "p_medium", "p_high"]].to_numpy()
        )
        authority = expected[model]
        if abs(result["macro_f1"] - authority["macro_f1"]) > 1e-9:
            raise RuntimeError(f"{dataset}/{model} frozen metric replay mismatch")
        rows.append(
            {
                "dataset": dataset,
                "task": "MAIN",
                "stage": "MAIN_ENDPOINT",
                "model": model,
                **{key: result[key] for key in METRICS},
            }
        )
        matrices[model] = result["confusion_matrix"]
        class_rows.extend(
            {"dataset": dataset, "task": "MAIN", "model": model, **item}
            for item in result["per_class"]
        )
    result_frame = pd.DataFrame(rows).sort_values("model").reset_index(drop=True)
    if set(result_frame.model) != set(PRIMARY):
        raise RuntimeError(f"{dataset} does not have eight primary models")
    return result_frame, pd.DataFrame(class_rows), matrices


def _uci_manifests(authority: dict[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for dataset, key in (("student_mat", "student_mat"), ("student_por", "student_por")):
        item = authority["uci"][key]
        config = ROOT / item["configuration"]
        metric = ROOT / item["metric_artifact"]
        checkpoint_dir = ROOT / item["checkpoint_directory"]
        prediction = ROOT / item["prediction_artifact"]
        result[dataset] = {
            "authority": item,
            "architecture_config": yaml.safe_load(config.read_text(encoding="utf-8")),
            "metric": _json(metric),
            "prediction_sha256": _hash(prediction),
            "metric_sha256": _hash(metric),
            "configuration_sha256": _hash(config),
            "checkpoint_count": len(list(checkpoint_dir.glob("*.pt"))),
            "checkpoint_sha256": {
                path.name: _hash(path) for path in sorted(checkpoint_dir.glob("*.pt"))
            },
            "protocol_snapshot": _json(ROOT / f"artifacts/final/protocol_snapshots/cnn_bilstm_{key.split('_')[1]}.json"),
        }
    return result


def _write_reports(
    authority: dict[str, Any],
    uci_main: pd.DataFrame,
    uci_stage: pd.DataFrame,
    uci_class: pd.DataFrame,
    oulad: pd.DataFrame,
    matrices: dict[str, Any],
    manifests: dict[str, Any],
) -> None:
    REPORT.mkdir(parents=True, exist_ok=True)
    main = pd.DataFrame(
        [
            ["Student-Mat", "CNN-BiLSTM frozen UCI", authority["uci"]["student_mat"]["macro_f1"]],
            ["Student-Por", "CNN-BiLSTM frozen UCI", authority["uci"]["student_por"]["macro_f1"]],
            ["OULAD FINAL", "H1 Tabular Residual CNN-BiLSTM", authority["oulad"]["final"]["macro_f1"]],
        ],
        columns=["Dataset", "Hybrid Architecture", "Macro-F1"],
    )
    (REPORT / "01_FINAL_MAIN_RESULTS.md").write_text(
        "# FINAL AUTHORITY — Main thesis results\n\n" + _markdown_table(main, list(main.columns)) + "\n",
        encoding="utf-8",
    )
    tables = []
    for dataset, frame in uci_main.groupby("dataset"):
        tables.append(f"## {dataset} MAIN — frozen UCI authority\n\n" + _markdown_table(frame.sort_values("macro_f1", ascending=False), ["model", *METRICS]))
    final_oulad = oulad.loc[oulad.stage.eq("FINAL")].copy()
    tables.append("## OULAD FINAL — Canonical V3\n\n" + _markdown_table(final_oulad.sort_values("macro_f1", ascending=False), ["model", *METRICS, "risk_precision", "risk_recall", "risk_f1", "specificity"]))
    (REPORT / "02_FULL_ML_VS_HYBRID.md").write_text("# FINAL AUTHORITY — ML vs Hybrid\n\n" + "\n\n".join(tables) + "\n", encoding="utf-8")
    stages = []
    for (dataset, stage), frame in uci_stage.groupby(["dataset", "stage"]):
        stages.append(f"## {dataset} {stage} — CANONICAL V3 secondary stage evidence\n\n" + _markdown_table(frame.sort_values("macro_f1", ascending=False), ["model", *METRICS]))
    (REPORT / "03_UCI_STAGE_RESULTS.md").write_text("# CANONICAL V3 — UCI secondary stages\n\n" + "\n\n".join(stages) + "\n", encoding="utf-8")
    stage_tables = []
    for stage, frame in oulad.groupby("stage"):
        stage_tables.append(f"## {stage}\n\n" + _markdown_table(frame.sort_values("macro_f1", ascending=False), ["model", *METRICS, "risk_precision", "risk_recall", "risk_f1", "specificity", "tp", "fp", "tn", "fn"]))
    (REPORT / "04_OULAD_STAGE_RESULTS.md").write_text("# CANONICAL V3 — OULAD stages and FINAL\n\n" + "\n\n".join(stage_tables) + "\n", encoding="utf-8")
    (REPORT / "05_FULL_METRICS.md").write_text(
        "# FINAL AUTHORITY — Full metrics\n\n"
        + "UCI main metrics are replayed from the frozen endpoint probabilities. UCI stage and OULAD metrics are replayed from their immutable canonical evidence.\n\n"
        + _markdown_table(uci_main, ["dataset", "model", *METRICS])
        + "\n\n"
        + _markdown_table(oulad, ["stage", "model", *METRICS, "risk_precision", "risk_recall", "risk_f1", "not_risk_precision", "not_risk_recall", "not_risk_f1", "specificity", "tp", "fp", "tn", "fn"])
        + "\n\n"
        + _markdown_table(uci_class, ["dataset", "task", "model", "class_name", "precision", "recall", "f1", "support"])
        + "\n",
        encoding="utf-8",
    )
    calibration = pd.concat([uci_main, oulad], ignore_index=True)
    (REPORT / "06_CALIBRATION_RESULTS.md").write_text("# FINAL AUTHORITY — Calibration\n\n" + _markdown_table(calibration, ["dataset", "task", "stage", "model", "nll", "brier", "ece"]) + "\n", encoding="utf-8")
    (REPORT / "07_CONFUSION_MATRICES.md").write_text("# FINAL AUTHORITY — Confusion matrices\n\n```json\n" + json.dumps(matrices, indent=2) + "\n```\n", encoding="utf-8")
    stats = _json(ROOT / "artifacts/canonical_v3/statistical_comparison.json")
    uci_stats = {
        dataset: pd.read_csv(ROOT / f"artifacts/final/comparator_completion/{dataset}/bootstrap_comparison.csv").to_dict(orient="records")
        for dataset in ("student_mat", "student_por")
    }
    (REPORT / "08_STATISTICAL_COMPARISON.md").write_text("# FINAL AUTHORITY — Paired statistical comparisons\n\n```json\n" + json.dumps({"uci_frozen": uci_stats, "oulad_canonical_v3": stats}, indent=2) + "\n```\n", encoding="utf-8")
    (REPORT / "09_HYBRID_STRENGTHS_WEAKNESSES.md").write_text(
        "# FINAL AUTHORITY — Hybrid strengths and weaknesses\n\n"
        "- The frozen UCI Hybrid is competitive but does not uniformly exceed the best tabular comparator at the two main endpoints.\n"
        "- H1 is strongest on canonical OULAD: Macro-F1 rises from 0.852491 at 75% to 0.894071 at FINAL, where it is a practical tie with MLP.\n"
        "- Phase 5 ablations are DEVELOPMENT evidence only: full 0.796611, residual-disabled 0.789914, temporal-disabled 0.784728. They are not Canonical V3 FINAL metrics.\n",
        encoding="utf-8",
    )
    (REPORT / "10_FINAL_MODEL_ARCHITECTURES.md").write_text("# FINAL AUTHORITY — Architecture registry\n\n```json\n" + json.dumps({"uci": manifests, "oulad": _json(ROOT / "artifacts/canonical_v3/CANONICAL_BENCHMARK_FREEZE.json")}, indent=2) + "\n```\n", encoding="utf-8")
    (REPORT / "11_FINAL_PIPELINES.md").write_text(
        "# FINAL AUTHORITY — Pipelines\n\n"
        "UCI uses its frozen V5.1 train-only preprocessing, five-fold and fixed-seed probability ensemble. OULAD uses Canonical V3 STRICT_REAL_TIME: score values and score-derived aggregates are excluded, while all canonical cutoff feature sets are monotonic.\n",
        encoding="utf-8",
    )
    (REPORT / "12_PROVENANCE.md").write_text("# FINAL AUTHORITY — Provenance\n\n```json\n" + json.dumps(manifests, indent=2) + "\n```\n", encoding="utf-8")


def main() -> int:
    authority = yaml.safe_load(AUTHORITY_PATH.read_text(encoding="utf-8"))
    RELEASE.mkdir(parents=True, exist_ok=True)
    mat, mat_class, mat_matrices = _uci_main("student_mat")
    por, por_class, por_matrices = _uci_main("student_por")
    uci_main = pd.concat([mat, por], ignore_index=True)
    if tuple(uci_main.loc[uci_main.model.eq("cnn_bilstm"), "macro_f1"]) != (
        authority["uci"]["student_mat"]["macro_f1"], authority["uci"]["student_por"]["macro_f1"]
    ):
        raise RuntimeError("frozen UCI authority values changed")
    uci_stage = pd.read_csv(ROOT / "artifacts/canonical_v3/uci_full_metrics_aggregate.csv")
    uci_stage = uci_stage.loc[uci_stage.task.eq("EARLY_WARNING")].copy()
    uci_stage["model"] = uci_stage.model.replace({"hybrid": "cnn_bilstm"})
    uci_class = pd.concat([mat_class, por_class], ignore_index=True)
    stage_class = pd.read_csv(ROOT / "artifacts/canonical_v3/per_class_metrics.csv")
    stage_class = stage_class.loc[
        stage_class.dataset.isin(["student_mat", "student_por"])
        & stage_class.task.eq("EARLY_WARNING")
    ].copy()
    oulad = pd.read_csv(ROOT / "artifacts/canonical_v3/oulad_full_metrics_aggregate.csv")
    oulad["model"] = oulad.model.replace({"hybrid": "h1_tabular_residual_expert"})
    if abs(oulad.loc[oulad.stage.eq("FINAL") & oulad.model.eq("h1_tabular_residual_expert"), "macro_f1"].iloc[0] - authority["oulad"]["final"]["macro_f1"]) > 1e-12:
        raise RuntimeError("Canonical OULAD authority value changed")
    manifests = _uci_manifests(authority)
    matrices = {"student_mat": mat_matrices, "student_por": por_matrices, "oulad": _json(ROOT / "artifacts/canonical_v3/confusion_matrices.json")}
    _write_csv(RELEASE / "uci_main_full_metrics.csv", uci_main)
    _write_csv(RELEASE / "uci_main_per_class_metrics.csv", uci_class)
    _write_csv(RELEASE / "uci_stage_full_metrics.csv", uci_stage)
    _write_csv(RELEASE / "uci_stage_per_class_metrics.csv", stage_class)
    _write_csv(RELEASE / "oulad_canonical_v3_full_metrics.csv", oulad)
    _write_json(RELEASE / "frozen_uci_authority_manifests.json", manifests)
    _write_json(RELEASE / "final_model_registry.json", authority)
    _write_json(RELEASE / "final_confusion_matrices.json", matrices)
    _write_reports(authority, uci_main, uci_stage, uci_class, oulad, matrices, manifests)
    _write_json(
        RELEASE / "FINAL_REPLAY_PASS.json",
        {"status": "FINAL_REPLAY_PASS", "training_performed": False, "primary_models": 8},
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
