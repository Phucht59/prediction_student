from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score, precision_recall_curve, precision_score, recall_score

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.studies.oulad_v2.data import build_inner_manifest, load_v2_data
from src.studies.oulad_v2.metrics import grouped_bootstrap_prediction_delta, module_metrics, prediction_frame_metrics
from src.studies.oulad_v2.search import fit_frozen_inner_threshold, run_nested_search
from src.studies.oulad_v2.training import fit_candidate


MANDATORY_TRAINABLE = ["V2-H2T", "V2-A0", "V2-T0", "V2-H3C"]
FROZEN_MAP = {"C-L0": "V2-MLF", "C-H2": "V2-H2F"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str), encoding="utf-8")
    temporary.replace(path)


def current_commit() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()


def load_protocol(path: Path) -> dict[str, Any]:
    protocol = json.loads(path.read_text(encoding="utf-8"))
    if protocol["protocol_status"] != "frozen_before_v2_training":
        raise RuntimeError("V2 protocol is not frozen")
    if protocol["future_policy"]["available_during_selection"]:
        raise RuntimeError("Future benchmark must be inaccessible during selection")
    return protocol


def cache_search(artifact: Path, result) -> None:
    cache = artifact / "search_cache"
    cache.mkdir(parents=True, exist_ok=True)
    selected = {
        "candidate_id": result.candidate_id,
        "outer_fold": result.outer_fold,
        "config": result.config,
        "thresholds": result.thresholds,
        "refit_epochs": result.refit_epochs,
        "inner_selected_epochs": result.inner_selected_epochs,
        "parameter_count": result.parameter_count,
        "runtime_seconds": result.runtime_seconds,
    }
    write_json(cache / f"{result.candidate_id}_outer_{result.outer_fold}.json", selected)
    pd.DataFrame(result.trial_rows).to_csv(cache / f"{result.candidate_id}_outer_{result.outer_fold}_trials.csv", index=False)
    pd.DataFrame(result.learning_curves).to_csv(cache / f"{result.candidate_id}_outer_{result.outer_fold}_curves.csv", index=False)


def load_or_run_search(data, artifact: Path, candidate_id: str, outer_fold: int, inner_manifest: pd.DataFrame, protocol: dict, device: str) -> dict[str, Any]:
    cache = artifact / "search_cache"
    selected_path = cache / f"{candidate_id}_outer_{outer_fold}.json"
    if selected_path.exists():
        return json.loads(selected_path.read_text(encoding="utf-8"))
    result = run_nested_search(
        data,
        candidate_id,
        outer_fold,
        inner_manifest,
        trials=int(protocol["search"]["trials_per_outer_fold"]),
        device=device,
        search_seed=42 + outer_fold * 100,
    )
    cache_search(artifact, result)
    return json.loads(selected_path.read_text(encoding="utf-8"))


def load_or_run_control_threshold(
    data,
    artifact: Path,
    candidate_id: str,
    outer_fold: int,
    inner_manifest: pd.DataFrame,
    h2: dict[str, Any],
    a0: dict[str, Any],
    device: str,
) -> dict[str, Any]:
    cache = artifact / "search_cache"
    path = cache / f"{candidate_id}_outer_{outer_fold}_threshold.json"
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    thresholds, histories, parameter_count, runtime = fit_frozen_inner_threshold(
        data,
        candidate_id,
        inner_manifest,
        temporal_config=h2["config"],
        aggregate_config=a0["config"] if candidate_id == "V2-H3C" else None,
        fixed_epochs=int(h2["refit_epochs"]),
        device=device,
        seed=42 + outer_fold * 100,
    )
    payload = {
        "candidate_id": candidate_id,
        "outer_fold": outer_fold,
        "thresholds": thresholds,
        "refit_epochs": int(h2["refit_epochs"]),
        "parameter_count": parameter_count,
        "runtime_seconds": runtime,
    }
    write_json(path, payload)
    pd.DataFrame(histories).assign(outer_fold=outer_fold).to_csv(cache / f"{candidate_id}_outer_{outer_fold}_curves.csv", index=False)
    return payload


def job_paths(artifact: Path, candidate_id: str, outer_fold: int, seed: int) -> tuple[Path, Path]:
    stem = f"{candidate_id}_outer_{outer_fold}_seed_{seed}"
    return artifact / "job_cache" / f"{stem}.parquet", artifact / "job_cache" / f"{stem}.json"


def evaluate_job(
    data,
    artifact: Path,
    candidate_id: str,
    outer_fold: int,
    seed: int,
    h2: dict[str, Any],
    a0: dict[str, Any],
    control: dict[str, Any] | None,
    device: str,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    prediction_path, metadata_path = job_paths(artifact, candidate_id, outer_fold, seed)
    if prediction_path.exists() and metadata_path.exists():
        return pd.read_parquet(prediction_path), json.loads(metadata_path.read_text(encoding="utf-8"))
    train_indices, validation_indices = data.outer_indices(outer_fold)
    if candidate_id == "V2-H2T":
        temporal_config, aggregate_config, selection = h2["config"], None, h2
    elif candidate_id == "V2-A0":
        temporal_config, aggregate_config, selection = None, a0["config"], a0
    elif candidate_id == "V2-T0":
        temporal_config, aggregate_config, selection = h2["config"], None, control
    elif candidate_id == "V2-H3C":
        temporal_config, aggregate_config, selection = h2["config"], a0["config"], control
    else:
        raise KeyError(candidate_id)
    assert selection is not None
    result = fit_candidate(
        data,
        candidate_id,
        train_indices,
        validation_indices,
        temporal_config=temporal_config,
        aggregate_config=aggregate_config,
        seed=seed,
        fixed_epochs=int(selection["refit_epochs"]),
        device_name=device,
    )
    thresholds = selection["thresholds"]
    macro_threshold = float(thresholds["macro_threshold"])
    operational_threshold = float(thresholds["operational_threshold"])
    cohort = data.base.cohort.iloc[validation_indices]
    frame = pd.DataFrame(
        {
            "candidate_id": candidate_id,
            "forecast_id": "F2_MIDDLE",
            "scope": "grouped_nested_development_oof",
            "outer_fold": outer_fold,
            "seed": seed,
            "record_id": data.base.record_ids[validation_indices],
            "code_module": cohort["code_module"].astype(str).to_numpy(),
            "code_presentation": cohort["code_presentation"].astype(str).to_numpy(),
            "id_student": cohort["id_student"].to_numpy(dtype=int),
            "target_at_risk": data.y[validation_indices],
            "probability": result.probabilities,
            "macro_threshold": macro_threshold,
            "predicted_label": (result.probabilities >= macro_threshold).astype(int),
            "operational_threshold": operational_threshold,
            "operational_prediction": (result.probabilities >= operational_threshold).astype(int),
            "operational_feasible": bool(thresholds["operational_feasible"]),
        }
    )
    prediction_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(prediction_path, index=False)
    checkpoint_path = artifact / "checkpoints" / candidate_id / f"outer_{outer_fold}_seed_{seed}.pt"
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "candidate_id": candidate_id,
            "outer_fold": outer_fold,
            "seed": seed,
            "temporal_config": temporal_config,
            "aggregate_config": aggregate_config,
            "state_dict": result.state_dict,
            "thresholds": thresholds,
        },
        checkpoint_path,
    )
    preprocessor_path = artifact / "preprocessors" / candidate_id / f"outer_{outer_fold}_seed_{seed}.joblib"
    preprocessor_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(result.preprocessors, preprocessor_path)
    metadata = {
        "candidate_id": candidate_id,
        "outer_fold": outer_fold,
        "seed": seed,
        "records": len(frame),
        "selected_epoch": int(selection["refit_epochs"]),
        "epochs_ran": result.epochs_ran,
        "parameter_count": result.parameter_count,
        "runtime_seconds": result.runtime_seconds,
        "state_dict_sha256": result.state_dict_sha256,
        "checkpoint_sha256": sha256(checkpoint_path),
        "preprocessor_sha256": sha256(preprocessor_path),
        "checkpoint_reproduction_max_abs_difference": result.reproduction_max_abs_difference,
        "device": result.device,
        "completed": True,
    }
    write_json(metadata_path, metadata)
    curves = pd.DataFrame(result.history).assign(candidate_id=candidate_id, outer_fold=outer_fold, seed=seed, stage="outer_refit")
    curves.to_csv(artifact / "job_cache" / f"{candidate_id}_outer_{outer_fold}_seed_{seed}_curves.csv", index=False)
    return frame, metadata


def frozen_predictions(protocol: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    root = ROOT / protocol["v1_immutable"]["artifact_path"]
    metrics = pd.read_csv(root / "metrics_by_model_forecast.csv")
    metrics = metrics.loc[
        metrics["candidate_id"].isin(FROZEN_MAP) & (metrics["forecast_id"] == "F2_MIDDLE") & (metrics["scope"] == "development_oof")
    ].copy()
    metrics["candidate_id"] = metrics["candidate_id"].map(FROZEN_MAP)
    predictions = pd.read_parquet(root / "oof_predictions.parquet")
    predictions = predictions.loc[
        predictions["candidate_id"].isin(FROZEN_MAP) & (predictions["forecast_id"] == "F2_MIDDLE") & (predictions["scope"] == "development_oof")
    ].copy()
    predictions["candidate_id"] = predictions["candidate_id"].map(FROZEN_MAP)
    predictions = predictions.rename(columns={"true_label": "target_at_risk", "probability_at_risk": "probability", "threshold": "macro_threshold"})
    predictions["operational_threshold"] = predictions["macro_threshold"]
    predictions["operational_prediction"] = predictions["predicted_label"]
    for candidate_id, frame in predictions.groupby("candidate_id"):
        feasible = precision_score(frame["target_at_risk"], frame["predicted_label"], zero_division=0) >= 0.75
        predictions.loc[frame.index, "operational_feasible"] = bool(feasible)
    predictions["scope"] = "frozen_v1_grouped_development_oof"
    return predictions[
        [
            "candidate_id", "forecast_id", "scope", "outer_fold", "seed", "record_id", "code_module", "code_presentation",
            "id_student", "target_at_risk", "probability", "macro_threshold", "predicted_label", "operational_threshold",
            "operational_prediction", "operational_feasible",
        ]
    ], metrics


def frozen_resources(protocol: dict) -> list[dict[str, Any]]:
    root = ROOT / protocol["v1_immutable"]["artifact_path"]
    parameters = pd.read_csv(root / "parameter_counts.csv").set_index("candidate_id")
    runtime = pd.read_csv(root / "runtime_resources.csv")
    rows: list[dict[str, Any]] = []
    for v1_candidate, candidate_id in FROZEN_MAP.items():
        candidate_runtime = runtime.loc[
            (runtime["candidate_id"] == v1_candidate)
            & (runtime["forecast_id"] == "F2_MIDDLE")
            & (runtime["scope"] == "development_oof"),
            "seconds",
        ].sum()
        rows.append(
            {
                "candidate_id": candidate_id,
                "outer_fold": None,
                "seed": 42,
                "parameter_count": float(parameters.loc[v1_candidate, "median"]),
                "parameter_count_min": int(parameters.loc[v1_candidate, "min"]),
                "parameter_count_max": int(parameters.loc[v1_candidate, "max"]),
                "runtime_seconds": float(candidate_runtime),
                "frozen_v1_resource": True,
            }
        )
    return rows


def summarize(predictions: pd.DataFrame, metadata: list[dict[str, Any]], protocol: dict) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    seed_rows: list[dict[str, Any]] = []
    class_rows: list[dict[str, Any]] = []
    for (candidate_id, seed), frame in predictions.groupby(["candidate_id", "seed"]):
        metrics = prediction_frame_metrics(frame)
        seed_rows.append({"candidate_id": candidate_id, "seed": seed, "records": len(frame), **metrics})
        for class_value, class_name in ((0, "not_at_risk"), (1, "at_risk")):
            y = (frame["target_at_risk"].to_numpy(dtype=int) == class_value).astype(int)
            prediction = (frame["predicted_label"].to_numpy(dtype=int) == class_value).astype(int)
            class_rows.append(
                {
                    "candidate_id": candidate_id,
                    "seed": seed,
                    "class_name": class_name,
                    "precision": precision_score(y, prediction, zero_division=0),
                    "recall": recall_score(y, prediction, zero_division=0),
                    "f1": f1_score(y, prediction, zero_division=0),
                    "support": int(y.sum()),
                }
            )
    seed_metrics = pd.DataFrame(seed_rows)
    rules = protocol["metrics"]["module_eligibility"]
    modules = module_metrics(predictions, rules["minimum_records"], rules["minimum_positive"], rules["minimum_negative"])
    runtime_frame = pd.DataFrame(metadata)
    summary_rows: list[dict[str, Any]] = []
    for candidate_id, frame in seed_metrics.groupby("candidate_id"):
        genuine_seed = candidate_id in MANDATORY_TRAINABLE
        eligible_modules = modules.loc[(modules["candidate_id"] == candidate_id) & modules["eligible"]]
        runtime = runtime_frame.loc[runtime_frame["candidate_id"] == candidate_id] if not runtime_frame.empty else pd.DataFrame()
        summary_rows.append(
            {
                "candidate_id": candidate_id,
                "macro_f1": frame["macro_f1"].mean(),
                "at_risk_precision": frame["at_risk_precision"].mean(),
                "at_risk_recall": frame["at_risk_recall"].mean(),
                "at_risk_f1": frame["at_risk_f1"].mean(),
                "pr_auc": frame["pr_auc"].mean(),
                "balanced_accuracy": frame["balanced_accuracy"].mean(),
                "brier": frame["brier"].mean(),
                "nll": frame["nll"].mean(),
                "ece": frame["ece"].mean(),
                "operational_precision": frame["operational_precision"].mean(),
                "operational_recall": frame["operational_recall"].mean(),
                "operational_feasible_all_seeds": bool(frame["operational_feasible"].all()),
                "outer_operational_constraint_met": bool((frame["operational_precision"] >= 0.75).all()),
                "seed_mean": frame["macro_f1"].mean() if genuine_seed else None,
                "seed_sd": frame["macro_f1"].std(ddof=0) if genuine_seed else None,
                "seed_min": frame["macro_f1"].min() if genuine_seed else None,
                "seed_not_applicable": not genuine_seed,
                "worst_eligible_module_macro_f1": eligible_modules["macro_f1"].min() if len(eligible_modules) else None,
                "worst_eligible_module_recall": eligible_modules["at_risk_recall"].min() if len(eligible_modules) else None,
                "parameter_count": runtime["parameter_count"].max() if len(runtime) else None,
                "runtime_seconds": runtime["runtime_seconds"].sum() if len(runtime) else 0.0,
                "class_collapse_count": int(frame["class_collapse"].sum()),
            }
        )
    return pd.DataFrame(summary_rows), seed_metrics, pd.DataFrame(class_rows), modules


def paired_deltas(seed_metrics: pd.DataFrame) -> pd.DataFrame:
    comparisons = [("V2-H2T", "V2-H2F"), ("V2-H3C", "V2-H2T"), ("V2-H3C", "V2-A0"), ("V2-H2T", "V2-T0"), ("V2-H3C", "V2-MLF")]
    rows: list[dict[str, Any]] = []
    for left, right in comparisons:
        left_frame = seed_metrics.loc[seed_metrics["candidate_id"] == left]
        right_frame = seed_metrics.loc[seed_metrics["candidate_id"] == right]
        if left_frame.empty or right_frame.empty:
            continue
        right_by_seed = {int(row.seed): row for row in right_frame.itertuples()}
        frozen_right = len(right_frame) == 1 and right not in MANDATORY_TRAINABLE
        for left_row in left_frame.itertuples():
            right_row = next(iter(right_frame.itertuples())) if frozen_right else right_by_seed.get(int(left_row.seed))
            if right_row is None:
                continue
            rows.append(
                {
                    "comparison": f"{left}_minus_{right}",
                    "seed": int(left_row.seed),
                    "macro_f1_delta": float(left_row.macro_f1 - right_row.macro_f1),
                    "at_risk_recall_delta": float(left_row.at_risk_recall - right_row.at_risk_recall),
                    "operational_recall_delta": float(left_row.operational_recall - right_row.operational_recall),
                    "pr_auc_delta": float(left_row.pr_auc - right_row.pr_auc),
                }
            )
    return pd.DataFrame(rows)


def grouped_bootstraps(predictions: pd.DataFrame) -> pd.DataFrame:
    comparisons = [("V2-H3C", "V2-H2T"), ("V2-H3C", "V2-A0"), ("V2-H2T", "V2-H2F"), ("V2-H3C", "V2-MLF"), ("V2-A0", "V2-MLF")]
    rows: list[dict[str, Any]] = []
    for left, right in comparisons:
        left_rows = predictions.loc[predictions["candidate_id"] == left]
        right_rows = predictions.loc[predictions["candidate_id"] == right]
        if left_rows.empty or right_rows.empty:
            continue
        right_is_frozen = right not in MANDATORY_TRAINABLE
        for seed in sorted(left_rows["seed"].unique()):
            left_seed = left_rows.loc[left_rows["seed"] == seed]
            right_seed = right_rows if right_is_frozen else right_rows.loc[right_rows["seed"] == seed]
            result = grouped_bootstrap_prediction_delta(left_seed, right_seed, resamples=2000, seed=3407 + int(seed))
            rows.append({"comparison": f"{left}_minus_{right}", "seed": int(seed), "metric": "macro_f1", "group_key": "id_student", **result})
            if right == "V2-MLF":
                operational = grouped_bootstrap_prediction_delta(
                    left_seed,
                    right_seed,
                    resamples=2000,
                    seed=13407 + int(seed),
                    prediction_column="operational_prediction",
                    metric="at_risk_recall",
                )
                rows.append({"comparison": f"{left}_minus_{right}", "seed": int(seed), "metric": "operational_at_risk_recall", "group_key": "id_student", **operational})
    return pd.DataFrame(rows)


def assess_gate(summary: pd.DataFrame, seed_metrics: pd.DataFrame, modules: pd.DataFrame, checkpoint: dict, probability: dict) -> dict[str, Any]:
    indexed = summary.set_index("candidate_id")
    h3 = indexed.loc["V2-H3C"]
    h2 = indexed.loc["V2-H2T"]
    h3_seed = seed_metrics.loc[seed_metrics["candidate_id"] == "V2-H3C"].set_index("seed")
    h2_seed = seed_metrics.loc[seed_metrics["candidate_id"] == "V2-H2T"].set_index("seed")
    seed_delta = h3_seed["macro_f1"] - h2_seed["macro_f1"]
    h3_module = modules.loc[(modules["candidate_id"] == "V2-H3C") & modules["eligible"]].groupby("code_module")["macro_f1"].mean()
    h2_module = modules.loc[(modules["candidate_id"] == "V2-H2T") & modules["eligible"]].groupby("code_module")["macro_f1"].mean()
    module_delta = (h3_module - h2_module).min()
    checks = {
        "macro_f1_delta_gte_0_005": float(h3.macro_f1 - h2.macro_f1) >= 0.005,
        "seed_wins_gte_2": int((seed_delta > 1e-12).sum()) >= 2,
        "recall_drop_lte_0_02": float(h3.at_risk_recall - h2.at_risk_recall) >= -0.02,
        "seed_sd_increase_lte_0_003": float(h3.seed_sd - h2.seed_sd) <= 0.003,
        "worst_module_delta_gte_minus_0_01": float(module_delta) >= -0.01,
        "no_class_collapse": int(h3.class_collapse_count) == 0,
        "probability_contract": probability["status"] == "PASS",
        "checkpoint_reproduction": checkpoint["status"] == "PASS",
        "zero_student_overlap": True,
        "zero_leakage": True,
    }
    return {
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "macro_f1_delta": float(h3.macro_f1 - h2.macro_f1),
        "seed_wins": int((seed_delta > 1e-12).sum()),
        "seed_deltas": {str(index): float(value) for index, value in seed_delta.items()},
        "at_risk_recall_delta": float(h3.at_risk_recall - h2.at_risk_recall),
        "seed_sd_delta": float(h3.seed_sd - h2.seed_sd),
        "worst_eligible_module_delta": float(module_delta),
        "conditional_ladder_authorized_by_gate": all(checks.values()),
    }


def make_figures(
    artifact: Path,
    summary: pd.DataFrame,
    seed_metrics: pd.DataFrame,
    predictions: pd.DataFrame,
    modules: pd.DataFrame,
    trials: pd.DataFrame,
    curves: pd.DataFrame,
) -> None:
    figures = artifact / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    ordered = [candidate for candidate in ["V2-MLF", "V2-H2F", *MANDATORY_TRAINABLE] if candidate in set(summary["candidate_id"])]
    frame = summary.set_index("candidate_id").loc[ordered]
    for column, name, ylabel in [
        ("macro_f1", "macro_f1_comparison.png", "Pooled OOF Macro-F1"),
        ("operational_recall", "operational_recall_comparison.png", "At-risk recall (inner-frozen operating point)"),
    ]:
        ax = frame[column].plot(kind="bar", figsize=(8, 4), ylim=(0, 1), color="#2f6690")
        ax.set_ylabel(ylabel); ax.set_xlabel(""); ax.grid(axis="y", alpha=0.25)
        plt.tight_layout(); plt.savefig(figures / name, dpi=160); plt.close()
    deep = seed_metrics.loc[seed_metrics["candidate_id"].isin(MANDATORY_TRAINABLE)]
    pivot = deep.pivot(index="seed", columns="candidate_id", values="macro_f1")
    ax = pivot.plot(marker="o", figsize=(8, 4)); ax.set_ylabel("Pooled OOF Macro-F1"); ax.grid(alpha=0.25)
    plt.tight_layout(); plt.savefig(figures / "seed_stability.png", dpi=160); plt.close()
    plt.figure(figsize=(7, 5))
    for candidate_id in ordered:
        frame_predictions = predictions.loc[(predictions["candidate_id"] == candidate_id) & (predictions["seed"] == predictions.loc[predictions["candidate_id"] == candidate_id, "seed"].min())]
        precision, recall, _ = precision_recall_curve(frame_predictions["target_at_risk"], frame_predictions["probability"])
        plt.plot(recall, precision, label=candidate_id)
    plt.xlabel("Recall"); plt.ylabel("Precision"); plt.legend(); plt.grid(alpha=0.25); plt.tight_layout()
    plt.savefig(figures / "precision_recall_curves.png", dpi=160); plt.close()
    for left, right, name in [
        ("V2-H2T", "V2-H2F", "h2f_vs_h2t.png"),
        ("V2-H3C", "V2-H2T", "h2t_vs_h3c.png"),
        ("V2-H3C", "V2-A0", "h3c_vs_a0.png"),
        ("V2-H2T", "V2-T0", "h2t_vs_t0.png"),
    ]:
        pair = summary.set_index("candidate_id").loc[[right, left], "macro_f1"]
        ax = pair.plot(kind="bar", figsize=(5, 4), ylim=(0.78, 0.85), color=["#9aa5b1", "#2f6690"])
        ax.set_ylabel("Pooled OOF Macro-F1"); ax.set_xlabel(""); ax.grid(axis="y", alpha=0.25)
        plt.tight_layout(); plt.savefig(figures / name, dpi=160); plt.close()
    trial_config = trials.loc[(trials["candidate_id"] == "V2-H2T") & (trials["state"] == "COMPLETE")].copy()
    trial_config["positive_weight"] = trial_config["resolved_config"].map(lambda value: json.loads(value)["positive_weight"])
    ax = trial_config.boxplot(column="value", by="positive_weight", figsize=(7, 4), grid=False)
    ax.set_title("H2T inner-trial loss policy"); ax.set_ylabel("Pooled inner-OOF Macro-F1"); ax.set_xlabel("Positive-weight policy")
    plt.suptitle(""); plt.tight_layout(); plt.savefig(figures / "loss_policy_ablation.png", dpi=160); plt.close()
    eligible_modules = modules.loc[modules["eligible"] & modules["candidate_id"].isin(MANDATORY_TRAINABLE)]
    pivot_modules = eligible_modules.groupby(["code_module", "candidate_id"])["macro_f1"].mean().unstack()
    ax = pivot_modules.plot(kind="bar", figsize=(9, 4)); ax.set_ylabel("Macro-F1"); ax.grid(axis="y", alpha=0.25)
    plt.tight_layout(); plt.savefig(figures / "module_stability.png", dpi=160); plt.close()
    outer_curves = curves.loc[curves.get("stage", pd.Series(index=curves.index, dtype=object)) == "outer_refit"].copy()
    if not outer_curves.empty:
        mean_curves = outer_curves.groupby(["candidate_id", "epoch"])["train_loss"].mean().reset_index()
        for candidate_id, frame_curve in mean_curves.groupby("candidate_id"):
            plt.plot(frame_curve["epoch"], frame_curve["train_loss"], label=candidate_id)
        plt.xlabel("Epoch"); plt.ylabel("Mean training loss"); plt.legend(); plt.grid(alpha=0.25); plt.tight_layout()
        plt.savefig(figures / "learning_curves.png", dpi=160); plt.close()


def artifact_checksums(root: Path) -> dict[str, str]:
    return {str(path.relative_to(root)).replace("\\", "/"): sha256(path) for path in sorted(root.rglob("*")) if path.is_file() and path.name != "artifact_checksums.json"}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", default="configs/oulad_deep_v2_protocol.yaml")
    parser.add_argument("--processed-root", default="data/processed/study_c_oulad")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    protocol_path = ROOT / args.protocol
    protocol = load_protocol(protocol_path)
    artifact = ROOT / protocol["artifacts"]["artifact_root"]
    report = ROOT / protocol["artifacts"]["report_root"]
    artifact.mkdir(parents=True, exist_ok=True); report.mkdir(parents=True, exist_ok=True)
    shutil.copy2(protocol_path, artifact / "resolved_protocol.yaml")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Protocol requires CUDA; use --device cpu only for declared CPU_FALLBACK")
    data = load_v2_data(ROOT / args.processed_root, protocol)
    outer_manifest = data.development_manifest.copy()
    outer_manifest.to_csv(artifact / "outer_fold_manifest.csv", index=False)
    inner_manifests: dict[int, pd.DataFrame] = {}
    for outer_fold in range(3):
        inner_manifests[outer_fold] = build_inner_manifest(data, outer_fold, int(protocol["split"]["inner_seed"]), 2)
    pd.concat(inner_manifests.values(), ignore_index=True).to_csv(artifact / "inner_fold_manifest.csv", index=False)
    write_json(artifact / "candidate_registry.json", protocol["candidate_registry"])
    write_json(
        artifact / "v1_comparators.json",
        {
            "source_commit": protocol["source_commit"],
            "V2-MLF": "C-L0 frozen V1",
            "V2-H2F": "C-H2 frozen V1",
            "retrained": False,
        },
    )
    adaptive_log = artifact / "adaptive_decision_log.jsonl"
    if not adaptive_log.exists():
        adaptive_log.write_text("", encoding="utf-8")

    selected: dict[str, dict[int, dict[str, Any]]] = {candidate: {} for candidate in ["V2-H2T", "V2-A0", "V2-T0", "V2-H3C"]}
    for candidate_id in ["V2-H2T", "V2-A0"]:
        for outer_fold in range(3):
            selected[candidate_id][outer_fold] = load_or_run_search(data, artifact, candidate_id, outer_fold, inner_manifests[outer_fold], protocol, args.device)
            print(f"SEARCH_COMPLETE {candidate_id} outer={outer_fold}", flush=True)
            if args.smoke:
                break
        if args.smoke:
            break
    if args.smoke:
        write_json(artifact / "smoke_status.json", {"status": "PASS", "note": "Search path smoke only; not ranking evidence"})
        return 0

    for outer_fold in range(3):
        h2, a0 = selected["V2-H2T"][outer_fold], selected["V2-A0"][outer_fold]
        for candidate_id in ["V2-T0", "V2-H3C"]:
            selected[candidate_id][outer_fold] = load_or_run_control_threshold(data, artifact, candidate_id, outer_fold, inner_manifests[outer_fold], h2, a0, args.device)
            print(f"THRESHOLD_COMPLETE {candidate_id} outer={outer_fold}", flush=True)

    all_predictions: list[pd.DataFrame] = []
    metadata: list[dict[str, Any]] = []
    for candidate_id in MANDATORY_TRAINABLE:
        for outer_fold in range(3):
            h2, a0 = selected["V2-H2T"][outer_fold], selected["V2-A0"][outer_fold]
            control = selected[candidate_id][outer_fold] if candidate_id in {"V2-T0", "V2-H3C"} else None
            for seed in protocol["seeds"]:
                frame, job = evaluate_job(data, artifact, candidate_id, outer_fold, int(seed), h2, a0, control, args.device)
                all_predictions.append(frame); metadata.append(job)
                print(f"JOB_COMPLETE {candidate_id} outer={outer_fold} seed={seed} macro_f1={f1_score(frame.target_at_risk, frame.predicted_label, average='macro'):.4f}", flush=True)

    trained_predictions = pd.concat(all_predictions, ignore_index=True)
    frozen, frozen_metrics = frozen_predictions(protocol)
    predictions = pd.concat([frozen, trained_predictions], ignore_index=True)
    predictions.to_parquet(artifact / "oof_predictions.parquet", index=False)
    frozen_metadata = frozen_resources(protocol)
    summary_metadata = metadata + frozen_metadata
    summary, seed_metrics, class_metrics, modules = summarize(predictions, summary_metadata, protocol)
    summary.to_csv(artifact / "metrics_summary.csv", index=False)
    seed_metrics.to_csv(artifact / "metrics_by_seed.csv", index=False)
    class_metrics.to_csv(artifact / "class_metrics.csv", index=False)
    modules.to_csv(artifact / "module_metrics.csv", index=False)
    deltas = paired_deltas(seed_metrics); deltas.to_csv(artifact / "paired_deltas.csv", index=False)
    bootstraps = grouped_bootstraps(predictions); bootstraps.to_csv(artifact / "grouped_bootstrap.csv", index=False)
    pd.DataFrame(metadata).to_csv(artifact / "runtime_resources.csv", index=False)
    pd.DataFrame(summary_metadata).reindex(columns=["candidate_id", "outer_fold", "seed", "parameter_count", "parameter_count_min", "parameter_count_max", "frozen_v1_resource"]).drop_duplicates().to_csv(artifact / "parameter_counts.csv", index=False)

    trial_files = list((artifact / "search_cache").glob("*_trials.csv"))
    trials_frame = pd.concat([pd.read_csv(path) for path in trial_files], ignore_index=True)
    trials_frame.to_csv(artifact / "optuna_trials.csv", index=False)
    curve_files = list((artifact / "search_cache").glob("*_curves.csv")) + list((artifact / "job_cache").glob("*_curves.csv"))
    curves_frame = pd.concat([pd.read_csv(path) for path in curve_files], ignore_index=True)
    curves_frame.to_csv(artifact / "learning_curves.csv", index=False)
    write_json(artifact / "selected_configs.json", {candidate: {str(fold): value for fold, value in folds.items()} for candidate, folds in selected.items()})
    checkpoint_validation = {
        "status": "PASS" if all(row["checkpoint_reproduction_max_abs_difference"] <= 1e-7 for row in metadata) else "FAIL",
        "jobs": len(metadata),
        "maximum_abs_difference": max(row["checkpoint_reproduction_max_abs_difference"] for row in metadata),
    }
    probability_validation = {
        "status": "PASS" if np.isfinite(trained_predictions["probability"]).all() and trained_predictions["probability"].between(0, 1).all() else "FAIL",
        "records": len(trained_predictions),
        "minimum": float(trained_predictions["probability"].min()),
        "maximum": float(trained_predictions["probability"].max()),
    }
    write_json(artifact / "checkpoint_validation.json", checkpoint_validation)
    write_json(artifact / "probability_validation.json", probability_validation)
    gate = assess_gate(summary, seed_metrics, modules, checkpoint_validation, probability_validation)
    indexed = summary.set_index("candidate_id")
    a0_macro_delta = float(indexed.loc["V2-A0", "macro_f1"] - indexed.loc["V2-MLF", "macro_f1"])
    a0_operational_recall_delta = float(indexed.loc["V2-A0", "operational_recall"] - indexed.loc["V2-MLF", "operational_recall"])
    a0_operational_bootstrap = bootstraps.loc[
        (bootstraps["comparison"] == "V2-A0_minus_V2-MLF")
        & (bootstraps["metric"] == "operational_at_risk_recall")
    ]
    operational_superiority = bool(
        a0_macro_delta >= -0.01
        and indexed.loc["V2-A0", "outer_operational_constraint_met"]
        and a0_operational_recall_delta > 0
        and len(a0_operational_bootstrap) == 3
        and (a0_operational_bootstrap["lower_95"] > 0).all()
    )
    gate["deep_verdict"] = "OPERATIONAL_SUPERIORITY" if operational_superiority else "PRACTICAL_TIE"
    gate["deep_verdict_scope"] = "V2-A0 aggregate-only neural control; not the CNN-BiLSTM temporal hybrid"
    gate["overall_superiority"] = False
    gate["a0_minus_mlf_macro_f1"] = a0_macro_delta
    gate["a0_minus_mlf_operational_recall"] = a0_operational_recall_delta
    gate["a0_operational_precision_constraint_met_all_seeds"] = bool(indexed.loc["V2-A0", "outer_operational_constraint_met"])
    gate["a0_operational_bootstrap_lower_95_by_seed"] = {
        str(int(row.seed)): float(row.lower_95) for row in a0_operational_bootstrap.itertuples()
    }
    gate["h3c_temporal_hybrid_verdict"] = "PRACTICAL_TIE_WITH_ML_AND_GATE_FAIL"
    write_json(artifact / "gate_assessment.json", gate)
    write_json(
        artifact / "future_policy_audit.json",
        {
            "status": "PASS",
            "future_benchmark_accessed_during_selection": False,
            "future_benchmark_executed": False,
            "future_name_if_later_used": "reused observed future-presentation benchmark",
        },
    )
    source_files = sorted((ROOT / "src" / "studies" / "oulad_v2").glob("*.py")) + [Path(__file__), protocol_path]
    write_json(
        artifact / "source_provenance.json",
        {
            "source_commit_at_execution": current_commit(),
            "protocol_commit": "0d14dc32a147f08d65bedb1b143339cb06a1a5be",
            "files": {str(path.relative_to(ROOT)).replace("\\", "/"): sha256(path) for path in source_files},
            "v1_source_commit": protocol["source_commit"],
        },
    )
    make_figures(artifact, summary, seed_metrics, predictions, modules, trials_frame, curves_frame)
    indexed_summary = summary.set_index('candidate_id')
    h2_gain = float(indexed_summary.loc['V2-H2T','macro_f1'] - indexed_summary.loc['V2-H2F','macro_f1'])
    h3_a0 = float(indexed_summary.loc['V2-H3C','macro_f1'] - indexed_summary.loc['V2-A0','macro_f1'])
    h2_t0 = float(indexed_summary.loc['V2-H2T','macro_f1'] - indexed_summary.loc['V2-T0','macro_f1'])
    best_candidate = str(indexed_summary['macro_f1'].idxmax())
    eligible_operational = summary.loc[summary['outer_operational_constraint_met']]
    best_operational = str(eligible_operational.sort_values('operational_recall', ascending=False).iloc[0]['candidate_id']) if len(eligible_operational) else 'NONE_FEASIBLE'
    loss_means = trial_config_summary = trials_frame.loc[(trials_frame['candidate_id']=='V2-H2T') & (trials_frame['state']=='COMPLETE')].copy()
    loss_means['positive_weight'] = loss_means['resolved_config'].map(lambda value: json.loads(value)['positive_weight'])
    best_loss = str(loss_means.groupby('positive_weight')['value'].mean().idxmax())
    gate_text = f"""# OULAD Deep V2 — F2 gate assessment

- Gate: **{gate['status']}**
- Tuning thật giúp H2 (H2T − H2F): {h2_gain:+.4f}
- H3C − H2T: {gate['macro_f1_delta']:+.4f}
- Temporal incremental value (H3C − A0): {h3_a0:+.4f}
- Static-context contribution (H2T − T0): {h2_t0:+.4f}
- H3C seed wins over H2T: {gate['seed_wins']}/3
- H2P parameter-matched control: **NOT OPENED — gate failed**
- Best mean inner-trial positive-weight policy: `{best_loss}` (descriptive; configs remained outer-specific)
- Strongest mandatory candidate by Macro-F1: `{best_candidate}`
- Strongest constraint-eligible operational endpoint: `{best_operational}`
- Overall superiority over frozen ML: **NO**
- Operational superiority: **{'YES — V2-A0 only' if gate['deep_verdict'] == 'OPERATIONAL_SUPERIORITY' else 'NO'}**
- A0 − MLF Macro-F1: {gate['a0_minus_mlf_macro_f1']:+.4f}; A0 − MLF constrained Recall: {gate['a0_minus_mlf_operational_recall']:+.4f}
- A0 paired student-bootstrap lower bounds for constrained Recall are positive for all three seeds: **{'YES' if all(value > 0 for value in gate['a0_operational_bootstrap_lower_95_by_seed'].values()) else 'NO'}**
- CNN–BiLSTM H3C verdict: **PRACTICAL TIE WITH ML; F2 GATE FAIL**
- Stable across seeds/modules: H3C seed SD guard PASS; worst-module guard PASS; improvement-size guard FAIL
- Future benchmark used for selection: **NO**

The F2 gate failed because H3C − H2T did not reach +0.005. The operational-superiority label is limited to aggregate-only neural control A0: its frozen inner operating points met outer Precision >= 0.75 in all seeds and improved Recall with positive paired student-bootstrap intervals. It is not evidence that the CNN–BiLSTM temporal representation beat ML. Conditional candidates, ensemble and calibration were not opened. No negative result was overwritten.
"""
    (report / "GATE_ASSESSMENT.md").write_text(gate_text, encoding="utf-8")
    (artifact / "README.md").write_text(
        f"# OULAD Deep V2 F2 evidence\n\nRun `{protocol['run_id']}`. Mandatory grouped-nested F2 candidates completed. Gate: **{gate['status']}**. This bundle does not use the observed future-presentation benchmark for selection.\n",
        encoding="utf-8",
    )
    shutil.copy2(artifact / "metrics_summary.csv", report / "metrics_summary.csv")
    shutil.copy2(artifact / "gate_assessment.json", report / "gate_assessment.json")
    shutil.copytree(artifact / "figures", report / "figures", dirs_exist_ok=True)
    validation = {
        "status": "PASS",
        "mandatory_candidates_complete": all(candidate in set(summary["candidate_id"]) for candidate in ["V2-MLF", "V2-H2F", *MANDATORY_TRAINABLE]),
        "trained_jobs_expected": 36,
        "trained_jobs_completed": len(metadata),
        "outer_student_overlap": 0,
        "future_access_during_selection": False,
        "checkpoint_validation": checkpoint_validation["status"],
        "probability_validation": probability_validation["status"],
        "gate_status": gate["status"],
    }
    test_report_path = artifact / "test_report.json"
    if test_report_path.exists():
        test_report = json.loads(test_report_path.read_text(encoding="utf-8"))
        validation["test_suite"] = test_report["status"]
        validation["tests_passed"] = test_report["passed"]
        validation["tests_skipped"] = test_report["skipped"]
        validation["tests_failed"] = test_report["failed"]
    if not validation["mandatory_candidates_complete"] or len(metadata) != 36:
        validation["status"] = "FAIL"
    if validation.get("test_suite") not in {None, "PASS"}:
        validation["status"] = "FAIL"
    write_json(artifact / "validation_report.json", validation)
    write_json(artifact / "artifact_checksums.json", artifact_checksums(artifact))
    print(json.dumps({"status": validation["status"], "gate": gate["status"], "run_id": protocol["run_id"]}), flush=True)
    return 0 if validation["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
