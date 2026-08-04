"""Nested OOF tuning for the final hybrid-only deterministic recommender.

This script searches only fixed arithmetic weights and abstention thresholds.
It never fits an auxiliary recommendation model.  Silver future labels are
visible only inside the appropriate training/validation partition and are not
inputs to the scorer.
"""
from __future__ import annotations

import hashlib
import itertools
import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "artifacts/recommend_hybrid/hybrid_only_final"
DATA = OUT / "dataset/candidate_rows.parquet"
PROTOCOL_PATH = ROOT / "configs/recommend_hybrid/hybrid_only_final_protocol.yaml"
sys.path.insert(0, str(ROOT))

RUNTIME_COLUMNS = [
    "risk_probability",
    "risk_uncertainty",
    "risk_reduction",
    "evidence_strength",
    "deficit_score",
    "workload_minutes",
    "action_available",
    "prerequisite_status",
]
EVALUATION_COLUMNS = ["silver_positive", "future_behavior_signal"]
SEED = 20260804


def _atomic_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _stable_inner_fold(value: str, salt: str, folds: int) -> int:
    digest = hashlib.sha256(f"{salt}|{value}".encode("utf-8")).hexdigest()
    return int(digest[:12], 16) % folds


def _quantile_scale(series: pd.Series, minimum: float) -> float:
    values = pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan)
    values = values.dropna()
    if values.empty:
        return minimum
    return max(minimum, float(values.clip(lower=0).quantile(0.95)))


def _fit_scales(frame: pd.DataFrame) -> dict[str, float]:
    return {
        "risk_scale": _quantile_scale(frame["risk_reduction"], 0.01),
        "need_scale": _quantile_scale(frame["deficit_score"], 0.01),
        "uncertainty_scale": _quantile_scale(frame["risk_uncertainty"], 0.01),
        "workload_scale_minutes": max(
            1.0, float(pd.to_numeric(frame["workload_minutes"], errors="coerce").max())
        ),
    }


def _weight_configs(protocol: dict[str, Any]) -> list[dict[str, float]]:
    space = protocol["search_space"]
    names = [
        "risk_weight",
        "evidence_weight",
        "need_weight",
        "certainty_weight",
        "workload_weight",
    ]
    return [
        {name: float(value) for name, value in zip(names, values, strict=True)}
        for values in itertools.product(*(space[name] for name in names))
    ]


def _threshold_configs(protocol: dict[str, Any]) -> list[dict[str, float]]:
    space = protocol["search_space"]
    names = [
        "minimum_risk_reduction",
        "maximum_uncertainty",
        "minimum_evidence",
        "minimum_top_margin",
        "minimum_top_score",
    ]
    return [
        {name: float(value) for name, value in zip(names, values, strict=True)}
        for values in itertools.product(*(space[name] for name in names))
    ]


def _score_rows(
    frame: pd.DataFrame,
    weights: dict[str, float],
    scales: dict[str, float],
) -> pd.Series:
    risk = (
        pd.to_numeric(frame["risk_reduction"], errors="coerce")
        .fillna(-np.inf)
        .clip(lower=0)
        .div(scales["risk_scale"])
        .clip(upper=1)
    )
    evidence = pd.to_numeric(frame["evidence_strength"], errors="coerce").fillna(0).clip(0, 1)
    need = (
        pd.to_numeric(frame["deficit_score"], errors="coerce")
        .fillna(0)
        .clip(lower=0)
        .div(scales["need_scale"])
        .clip(upper=1)
    )
    uncertainty = (
        pd.to_numeric(frame["risk_uncertainty"], errors="coerce")
        .fillna(1)
        .clip(lower=0)
    )
    certainty = 1.0 - uncertainty.div(scales["uncertainty_scale"]).clip(upper=1)
    workload = (
        pd.to_numeric(frame["workload_minutes"], errors="coerce")
        .fillna(scales["workload_scale_minutes"])
        .clip(lower=0)
        .div(scales["workload_scale_minutes"])
        .clip(upper=1)
    )
    return (
        weights["risk_weight"] * risk
        + weights["evidence_weight"] * evidence
        + weights["need_weight"] * need
        + weights["certainty_weight"] * certainty
        - weights["workload_weight"] * workload
    ).astype(float)


def _top_rows(frame: pd.DataFrame, scores: pd.Series) -> pd.DataFrame:
    working = frame.copy()
    working["hybrid_score"] = scores.to_numpy(dtype=float)
    working = working[
        (working["action_available"] == 1)
        & (working["prerequisite_status"] == 1)
        & np.isfinite(working["risk_reduction"])
    ].copy()
    working.sort_values(
        [
            "group_id",
            "hybrid_score",
            "risk_reduction",
            "risk_uncertainty",
            "workload_minutes",
            "runtime_action_id",
        ],
        ascending=[True, False, False, True, True, True],
        kind="stable",
        inplace=True,
    )
    working["rank"] = working.groupby("group_id", sort=False).cumcount()
    top = working[working["rank"] == 0].copy()
    second = (
        working[working["rank"] == 1][["group_id", "hybrid_score"]]
        .rename(columns={"hybrid_score": "second_score"})
    )
    top = top.merge(second, on="group_id", how="left")
    top["second_score"] = top["second_score"].fillna(0.0)
    top["top_margin"] = top["hybrid_score"] - top["second_score"]
    group_positive = (
        frame.groupby("group_id", sort=False)["silver_positive"].max().rename("group_has_positive")
    )
    top = top.merge(group_positive, on="group_id", how="left")
    return top


def _issued_mask(top: pd.DataFrame, thresholds: dict[str, float]) -> pd.Series:
    return (
        (top["risk_reduction"] >= thresholds["minimum_risk_reduction"])
        & (top["risk_uncertainty"] <= thresholds["maximum_uncertainty"])
        & (top["evidence_strength"] >= thresholds["minimum_evidence"])
        & (top["top_margin"] >= thresholds["minimum_top_margin"])
        & (top["hybrid_score"] >= thresholds["minimum_top_score"])
    )


def _metrics(top: pd.DataFrame, thresholds: dict[str, float]) -> dict[str, float | int]:
    issued = _issued_mask(top, thresholds)
    issued_rows = top[issued]
    correct = issued_rows["silver_positive"].astype(int)
    positive_groups = int(top["group_has_positive"].sum())
    issued_positive_groups = int((issued & (top["group_has_positive"] == 1)).sum())
    precision = float(correct.mean()) if len(correct) else 0.0
    coverage = float(issued_positive_groups / positive_groups) if positive_groups else 0.0
    correct_abstention = (~issued) & (top["group_has_positive"] == 0)
    selective_accuracy = float((correct.sum() + correct_abstention.sum()) / len(top)) if len(top) else 0.0

    action_precision = (
        issued_rows.groupby("action_family", observed=True)["silver_positive"].agg(["mean", "count"])
        if len(issued_rows)
        else pd.DataFrame(columns=["mean", "count"])
    )
    supported_action_precision = action_precision[action_precision["count"] >= 30]["mean"]
    macro_action_precision = (
        float(supported_action_precision.mean()) if len(supported_action_precision) else 0.0
    )
    stage_precision = (
        issued_rows.groupby("stage", observed=True)["silver_positive"].agg(["mean", "count"])
        if len(issued_rows)
        else pd.DataFrame(columns=["mean", "count"])
    )
    supported_stage_precision = stage_precision[stage_precision["count"] >= 50]["mean"]
    worst_stage_precision = (
        float(supported_stage_precision.min()) if len(supported_stage_precision) else 0.0
    )
    diversity = int(issued_rows["action_family"].nunique()) if len(issued_rows) else 0
    if len(issued_rows):
        concentration = float(issued_rows["action_family"].value_counts(normalize=True).max())
    else:
        concentration = 1.0
    return {
        "precision_at_1": precision,
        "actionable_coverage": coverage,
        "selective_accuracy": selective_accuracy,
        "macro_action_precision": macro_action_precision,
        "worst_stage_precision": worst_stage_precision,
        "action_diversity": diversity,
        "top_action_concentration": concentration,
        "issued_groups": int(issued.sum()),
        "total_groups": int(len(top)),
        "positive_groups": positive_groups,
    }


def _config_id(config: dict[str, float]) -> str:
    raw = json.dumps(config, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


def _select_from_training(
    frame: pd.DataFrame,
    protocol: dict[str, Any],
    *,
    salt: str,
) -> tuple[dict[str, float], pd.DataFrame]:
    inner_count = int(protocol["evaluation"]["inner_group_folds"])
    frame = frame.copy()
    frame["inner_fold"] = frame["base_record_id"].astype(str).map(
        lambda value: _stable_inner_fold(value, salt, inner_count)
    )
    weights = _weight_configs(protocol)
    thresholds = _threshold_configs(protocol)
    accumulator: dict[str, dict[str, Any]] = {}

    for inner_fold in range(inner_count):
        train = frame[frame["inner_fold"] != inner_fold].copy()
        validation = frame[frame["inner_fold"] == inner_fold].copy()
        if train.empty or validation.empty:
            raise RuntimeError(f"empty inner partition {inner_fold}")
        scales = _fit_scales(train)
        for weight_config in weights:
            top = _top_rows(validation, _score_rows(validation, weight_config, scales))
            for threshold_config in thresholds:
                full_config = {**weight_config, **threshold_config}
                config_id = _config_id(full_config)
                record = accumulator.setdefault(
                    config_id,
                    {"config_id": config_id, **full_config, "fold_metrics": []},
                )
                record["fold_metrics"].append(_metrics(top, threshold_config))

    rows: list[dict[str, Any]] = []
    target = float(protocol["selection_rule"]["target_precision"])
    coverage_floor = float(protocol["selection_rule"]["minimum_inner_coverage"])
    diversity_floor = int(protocol["selection_rule"]["minimum_action_diversity"])
    concentration_cap = float(
        protocol["selection_rule"]["maximum_top_action_concentration"]
    )
    for record in accumulator.values():
        fold_metrics = record.pop("fold_metrics")
        summary = {
            **record,
            "mean_precision_at_1": float(np.mean([m["precision_at_1"] for m in fold_metrics])),
            "mean_actionable_coverage": float(
                np.mean([m["actionable_coverage"] for m in fold_metrics])
            ),
            "mean_selective_accuracy": float(
                np.mean([m["selective_accuracy"] for m in fold_metrics])
            ),
            "mean_macro_action_precision": float(
                np.mean([m["macro_action_precision"] for m in fold_metrics])
            ),
            "worst_stage_precision": float(
                min(m["worst_stage_precision"] for m in fold_metrics)
            ),
            "minimum_action_diversity": int(
                min(m["action_diversity"] for m in fold_metrics)
            ),
            "maximum_top_action_concentration": float(
                max(m["top_action_concentration"] for m in fold_metrics)
            ),
        }
        summary["meets_target"] = bool(
            summary["mean_precision_at_1"] >= target
            and summary["mean_actionable_coverage"] >= coverage_floor
            and summary["minimum_action_diversity"] >= diversity_floor
            and summary["maximum_top_action_concentration"] <= concentration_cap
        )
        rows.append(summary)

    trials = pd.DataFrame(rows)
    eligible = trials[trials["meets_target"]].copy()
    if eligible.empty:
        eligible = trials[
            trials["mean_actionable_coverage"] >= coverage_floor
        ].copy()
    if eligible.empty:
        eligible = trials.copy()
    eligible.sort_values(
        [
            "meets_target",
            "mean_actionable_coverage",
            "worst_stage_precision",
            "mean_macro_action_precision",
            "mean_precision_at_1",
            "mean_selective_accuracy",
            "maximum_top_action_concentration",
            "config_id",
        ],
        ascending=[False, False, False, False, False, False, True, True],
        kind="stable",
        inplace=True,
    )
    selected_row = eligible.iloc[0]
    selected = {
        key: float(selected_row[key])
        for key in [
            "risk_weight",
            "evidence_weight",
            "need_weight",
            "certainty_weight",
            "workload_weight",
            "minimum_risk_reduction",
            "maximum_uncertainty",
            "minimum_evidence",
            "minimum_top_margin",
            "minimum_top_score",
        ]
    }
    selected["config_id"] = str(selected_row["config_id"])
    selected["inner_target_met"] = bool(selected_row["meets_target"])
    return selected, trials


def _evaluate_partition(
    train: pd.DataFrame,
    test: pd.DataFrame,
    selected: dict[str, float],
) -> tuple[pd.DataFrame, dict[str, Any], dict[str, float]]:
    scales = _fit_scales(train)
    weight_config = {
        key: float(selected[key])
        for key in [
            "risk_weight",
            "evidence_weight",
            "need_weight",
            "certainty_weight",
            "workload_weight",
        ]
    }
    threshold_config = {
        key: float(selected[key])
        for key in [
            "minimum_risk_reduction",
            "maximum_uncertainty",
            "minimum_evidence",
            "minimum_top_margin",
            "minimum_top_score",
        ]
    }
    top = _top_rows(test, _score_rows(test, weight_config, scales))
    top["issued"] = _issued_mask(top, threshold_config).astype(int)
    top["correct_top1"] = (
        (top["issued"] == 1) & (top["silver_positive"] == 1)
    ).astype(int)
    metrics = _metrics(top, threshold_config)
    metrics["selected_config_id"] = str(selected["config_id"])
    return top, metrics, scales


def _baseline_metrics(frame: pd.DataFrame) -> dict[str, dict[str, Any]]:
    results: dict[str, dict[str, Any]] = {}
    baseline_scores = {
        "risk_reduction_only": pd.to_numeric(frame["risk_reduction"], errors="coerce").fillna(-1e9),
        "evidence_only": pd.to_numeric(frame["evidence_strength"], errors="coerce").fillna(-1e9),
        "lowest_workload": -pd.to_numeric(frame["workload_minutes"], errors="coerce").fillna(1e9),
    }
    no_abstention = {
        "minimum_risk_reduction": -1e9,
        "maximum_uncertainty": 1.0,
        "minimum_evidence": 0.0,
        "minimum_top_margin": -1e9,
        "minimum_top_score": -1e9,
    }
    for name, scores in baseline_scores.items():
        results[name] = _metrics(_top_rows(frame, scores), no_abstention)
    return results


def main() -> None:
    protocol = yaml.safe_load(PROTOCOL_PATH.read_text(encoding="utf-8"))
    frame = pd.read_parquet(DATA)
    missing = set(RUNTIME_COLUMNS + EVALUATION_COLUMNS) - set(frame.columns)
    if missing:
        raise RuntimeError(f"hybrid-only candidate schema missing: {sorted(missing)}")
    if frame["base_record_id"].isna().any():
        raise RuntimeError("base_record_id is required for grouped evaluation")

    evaluation_out = OUT / "evaluation"
    model_selection_out = OUT / "model_selection"
    evaluation_out.mkdir(parents=True, exist_ok=True)
    model_selection_out.mkdir(parents=True, exist_ok=True)
    oof_predictions: list[pd.DataFrame] = []
    fold_metrics: list[dict[str, Any]] = []
    baseline_rows: list[dict[str, Any]] = []

    for outer_fold in protocol["evaluation"]["outer_folds"]:
        train = frame[frame["outer_fold"] != outer_fold].copy()
        test = frame[frame["outer_fold"] == outer_fold].copy()
        selected, trials = _select_from_training(
            train,
            protocol,
            salt=f"outer-{outer_fold}-{SEED}",
        )
        trials.to_csv(model_selection_out / f"fold_{outer_fold}_trials.csv", index=False)
        _atomic_json(model_selection_out / f"fold_{outer_fold}_selected.json", selected)
        predictions, metrics, scales = _evaluate_partition(train, test, selected)
        predictions["outer_fold"] = int(outer_fold)
        predictions.to_parquet(evaluation_out / f"fold_{outer_fold}_predictions.parquet", index=False)
        _atomic_json(
            evaluation_out / f"fold_{outer_fold}_metrics.json",
            {"metrics": metrics, "scales": scales, "selected": selected},
        )
        oof_predictions.append(predictions)
        fold_metrics.append({"outer_fold": int(outer_fold), **metrics})
        for name, baseline in _baseline_metrics(test).items():
            baseline_rows.append({"outer_fold": int(outer_fold), "method": name, **baseline})

    oof = pd.concat(oof_predictions, ignore_index=True)
    oof.to_parquet(evaluation_out / "OOF_PREDICTIONS.parquet", index=False)
    issued = oof[oof["issued"] == 1]
    positive_groups = int(oof["group_has_positive"].sum())
    overall = {
        "precision_at_1": float(issued["silver_positive"].mean()) if len(issued) else 0.0,
        "actionable_coverage": float(
            ((oof["issued"] == 1) & (oof["group_has_positive"] == 1)).sum()
            / positive_groups
        )
        if positive_groups
        else 0.0,
        "issued_groups": int(len(issued)),
        "total_groups": int(len(oof)),
        "positive_groups": positive_groups,
        "action_diversity": int(issued["action_family"].nunique()) if len(issued) else 0,
        "top_action_concentration": float(
            issued["action_family"].value_counts(normalize=True).max()
        )
        if len(issued)
        else 1.0,
    }
    pd.DataFrame(fold_metrics).to_csv(evaluation_out / "FOLD_METRICS.csv", index=False)
    pd.DataFrame(baseline_rows).to_csv(evaluation_out / "BASELINE_METRICS.csv", index=False)

    final_selected, final_trials = _select_from_training(
        frame,
        protocol,
        salt=f"final-runtime-{SEED}",
    )
    final_trials.to_csv(model_selection_out / "final_runtime_trials.csv", index=False)
    final_scales = _fit_scales(frame)
    _atomic_json(
        OUT / "HYBRID_ONLY_SELECTED_CONFIG.json",
        {
            "status": "SELECTED_NOT_RELEASED",
            "config": final_selected,
            "normalization_scales": final_scales,
            "additional_learned_model": False,
            "silver_labels_used_at_runtime": False,
            "claim_boundary": protocol["claim_boundary"],
        },
    )
    _atomic_json(
        evaluation_out / "OOF_RESULTS.json",
        {
            "status": "COMPLETE",
            "overall": overall,
            "folds": fold_metrics,
            "target_precision": protocol["release_gates"]["top1_precision_minimum"],
            "additional_learned_model": False,
            "future_features_in_scoring": False,
        },
    )
    print(json.dumps(overall, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
