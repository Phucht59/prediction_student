"""Fast nested OOF tuning for the deterministic hybrid-only recommender.

Candidates are filtered by risk, uncertainty, evidence, availability and
prerequisites before ranking, exactly as in the runtime scorer.  Each learner-
stage group is represented as a matrix with at most five scientific actions.
The script tunes arithmetic weights and abstention thresholds only; it does not
fit an auxiliary recommendation model.
"""
from __future__ import annotations

import hashlib
import itertools
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "artifacts/recommend_hybrid/hybrid_only_final"
DATA = OUT / "dataset/candidate_rows.parquet"
PROTOCOL_PATH = ROOT / "configs/recommend_hybrid/hybrid_only_final_protocol.yaml"
SEED = 20260804
ACTION_FAMILIES = (
    "ASSESSMENT_COMPLETION",
    "STUDY_REGULARITY",
    "VLE_ENGAGEMENT",
    "QUIZ_OR_RETRIEVAL_PRACTICE",
    "CONTENT_REVIEW",
)
RUNTIME_ACTIONS = (
    "ASSESSMENT_COMPLETION",
    "STUDY_SCHEDULE",
    "VLE_ENGAGEMENT",
    "RETRIEVAL_PRACTICE",
    "LEARNING_CONSOLIDATION",
)
ACTION_INDEX = {name: index for index, name in enumerate(ACTION_FAMILIES)}


@dataclass(frozen=True)
class GroupArrays:
    metadata: pd.DataFrame
    valid: np.ndarray
    risk_reduction: np.ndarray
    uncertainty: np.ndarray
    evidence: np.ndarray
    need: np.ndarray
    workload: np.ndarray
    silver: np.ndarray

    @property
    def group_count(self) -> int:
        return len(self.metadata)


@dataclass(frozen=True)
class TopSelection:
    exists: np.ndarray
    top_index: np.ndarray
    top_score: np.ndarray
    top_margin: np.ndarray
    silver: np.ndarray
    risk_reduction: np.ndarray
    uncertainty: np.ndarray
    evidence: np.ndarray
    workload: np.ndarray


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
    values = values.dropna().clip(lower=0)
    return max(minimum, float(values.quantile(0.95))) if len(values) else minimum


def _fit_scales(frame: pd.DataFrame) -> dict[str, float]:
    return {
        "risk_scale": _quantile_scale(frame["risk_reduction"], 0.01),
        "need_scale": _quantile_scale(frame["deficit_score"], 0.01),
        "uncertainty_scale": _quantile_scale(frame["risk_uncertainty"], 0.01),
        "workload_scale_minutes": max(
            1.0,
            float(pd.to_numeric(frame["workload_minutes"], errors="coerce").max()),
        ),
    }


def _group_arrays(frame: pd.DataFrame) -> GroupArrays:
    ordered = frame.sort_values(["group_id", "action_family"], kind="stable").copy()
    duplicate = ordered.duplicated(["group_id", "action_family"])
    if duplicate.any():
        raise RuntimeError("duplicate scientific action within a ranking group")
    metadata = (
        ordered.groupby("group_id", sort=False)
        .first()[
            ["base_record_id", "stage", "outer_fold", "course", "presentation"]
        ]
        .reset_index()
    )
    group_map = {value: index for index, value in enumerate(metadata["group_id"].astype(str))}
    shape = (len(metadata), len(ACTION_FAMILIES))
    valid = np.zeros(shape, dtype=bool)
    matrices = {
        "risk_reduction": np.full(shape, np.nan, dtype=np.float64),
        "uncertainty": np.full(shape, np.nan, dtype=np.float64),
        "evidence": np.full(shape, np.nan, dtype=np.float64),
        "need": np.full(shape, np.nan, dtype=np.float64),
        "workload": np.full(shape, np.nan, dtype=np.float64),
        "silver": np.zeros(shape, dtype=np.int8),
    }
    for row in ordered.itertuples():
        group_index = group_map[str(row.group_id)]
        action_index = ACTION_INDEX.get(str(row.action_family))
        if action_index is None:
            raise RuntimeError(f"unknown scientific action family: {row.action_family}")
        valid[group_index, action_index] = bool(
            int(row.action_available) == 1
            and int(row.prerequisite_status) == 1
        )
        matrices["risk_reduction"][group_index, action_index] = float(row.risk_reduction)
        matrices["uncertainty"][group_index, action_index] = float(row.risk_uncertainty)
        matrices["evidence"][group_index, action_index] = float(row.evidence_strength)
        matrices["need"][group_index, action_index] = float(row.deficit_score)
        matrices["workload"][group_index, action_index] = float(row.workload_minutes)
        matrices["silver"][group_index, action_index] = int(row.silver_positive)
    valid &= np.isfinite(matrices["risk_reduction"])
    valid &= np.isfinite(matrices["uncertainty"])
    valid &= np.isfinite(matrices["evidence"])
    valid &= np.isfinite(matrices["need"])
    valid &= np.isfinite(matrices["workload"])
    return GroupArrays(metadata=metadata, valid=valid, **matrices)


def _components(arrays: GroupArrays, scales: dict[str, float]) -> dict[str, np.ndarray]:
    return {
        "risk": np.clip(np.maximum(arrays.risk_reduction, 0.0) / scales["risk_scale"], 0, 1),
        "evidence": np.clip(arrays.evidence, 0, 1),
        "need": np.clip(np.maximum(arrays.need, 0.0) / scales["need_scale"], 0, 1),
        "certainty": 1.0 - np.clip(
            np.maximum(arrays.uncertainty, 0.0) / scales["uncertainty_scale"], 0, 1
        ),
        "workload": np.clip(
            np.maximum(arrays.workload, 0.0) / scales["workload_scale_minutes"], 0, 1
        ),
    }


def _score_matrix(components: dict[str, np.ndarray], weights: dict[str, float]) -> np.ndarray:
    return (
        weights["risk_weight"] * components["risk"]
        + weights["evidence_weight"] * components["evidence"]
        + weights["need_weight"] * components["need"]
        + weights["certainty_weight"] * components["certainty"]
        - weights["workload_weight"] * components["workload"]
    )


def _top_selection(
    arrays: GroupArrays,
    scores: np.ndarray,
    base_thresholds: dict[str, float],
) -> TopSelection:
    eligible = arrays.valid.copy()
    eligible &= arrays.risk_reduction >= base_thresholds["minimum_risk_reduction"]
    eligible &= arrays.uncertainty <= base_thresholds["maximum_uncertainty"]
    eligible &= arrays.evidence >= base_thresholds["minimum_evidence"]
    masked = np.where(eligible, scores, -np.inf)
    row_index = np.arange(arrays.group_count)
    top_index = np.argmax(masked, axis=1)
    top_score = masked[row_index, top_index]
    exists = np.isfinite(top_score)
    without_top = masked.copy()
    without_top[row_index, top_index] = -np.inf
    second = np.max(without_top, axis=1)
    second = np.where(np.isfinite(second), second, 0.0)
    margin = np.where(exists, top_score - second, 0.0)
    return TopSelection(
        exists=exists,
        top_index=top_index,
        top_score=np.where(exists, top_score, 0.0),
        top_margin=margin,
        silver=arrays.silver[row_index, top_index],
        risk_reduction=arrays.risk_reduction[row_index, top_index],
        uncertainty=arrays.uncertainty[row_index, top_index],
        evidence=arrays.evidence[row_index, top_index],
        workload=arrays.workload[row_index, top_index],
    )


def _metrics(
    arrays: GroupArrays,
    top: TopSelection,
    confidence: dict[str, float],
) -> dict[str, float | int]:
    issued = (
        top.exists
        & (top.top_score >= confidence["minimum_top_score"])
        & (top.top_margin >= confidence["minimum_top_margin"])
    )
    group_positive = arrays.silver.max(axis=1) > 0
    correct = issued & (top.silver == 1)
    precision = float(correct.sum() / issued.sum()) if issued.any() else 0.0
    coverage = (
        float((issued & group_positive).sum() / group_positive.sum())
        if group_positive.any()
        else 0.0
    )
    selective_accuracy = float(
        (correct.sum() + ((~issued) & (~group_positive)).sum()) / len(issued)
    ) if len(issued) else 0.0
    selected_actions = top.top_index[issued]
    diversity = int(np.unique(selected_actions).size) if issued.any() else 0
    if issued.any():
        counts = np.bincount(selected_actions, minlength=len(ACTION_FAMILIES))
        concentration = float(counts.max() / counts.sum())
    else:
        concentration = 1.0

    action_precisions = []
    for action_index in range(len(ACTION_FAMILIES)):
        mask = issued & (top.top_index == action_index)
        if mask.sum() >= 30:
            action_precisions.append(float(top.silver[mask].mean()))
    macro_action = float(np.mean(action_precisions)) if action_precisions else 0.0

    stages = arrays.metadata["stage"].astype(str).to_numpy()
    stage_precisions = []
    for stage in sorted(set(stages)):
        mask = issued & (stages == stage)
        if mask.sum() >= 50:
            stage_precisions.append(float(top.silver[mask].mean()))
    worst_stage = min(stage_precisions) if stage_precisions else 0.0
    return {
        "precision_at_1": precision,
        "actionable_coverage": coverage,
        "selective_accuracy": selective_accuracy,
        "macro_action_precision": macro_action,
        "worst_stage_precision": worst_stage,
        "action_diversity": diversity,
        "top_action_concentration": concentration,
        "issued_groups": int(issued.sum()),
        "total_groups": int(len(issued)),
        "positive_groups": int(group_positive.sum()),
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


def _base_thresholds(protocol: dict[str, Any]) -> list[dict[str, float]]:
    space = protocol["search_space"]
    names = ["minimum_risk_reduction", "maximum_uncertainty", "minimum_evidence"]
    return [
        {name: float(value) for name, value in zip(names, values, strict=True)}
        for values in itertools.product(*(space[name] for name in names))
    ]


def _confidence_thresholds(protocol: dict[str, Any]) -> list[dict[str, float]]:
    space = protocol["search_space"]
    return [
        {"minimum_top_margin": float(margin), "minimum_top_score": float(score)}
        for margin, score in itertools.product(
            space["minimum_top_margin"], space["minimum_top_score"]
        )
    ]


def _config_id(config: dict[str, float]) -> str:
    encoded = json.dumps(config, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:16]


def _shortlist_weights(
    fold_objects: list[tuple[GroupArrays, dict[str, float]]],
    weights: list[dict[str, float]],
    count: int,
) -> list[dict[str, float]]:
    permissive = {
        "minimum_risk_reduction": -1e9,
        "maximum_uncertainty": 1.0,
        "minimum_evidence": 0.0,
    }
    confidence = {"minimum_top_margin": -1e9, "minimum_top_score": -1e9}
    rows = []
    for weight in weights:
        fold_metrics = []
        for arrays, scales in fold_objects:
            top = _top_selection(
                arrays,
                _score_matrix(_components(arrays, scales), weight),
                permissive,
            )
            fold_metrics.append(_metrics(arrays, top, confidence))
        rows.append(
            {
                **weight,
                "mean_precision": float(np.mean([m["precision_at_1"] for m in fold_metrics])),
                "mean_coverage": float(np.mean([m["actionable_coverage"] for m in fold_metrics])),
                "worst_stage": float(min(m["worst_stage_precision"] for m in fold_metrics)),
                "minimum_diversity": int(min(m["action_diversity"] for m in fold_metrics)),
            }
        )
    ranking = pd.DataFrame(rows).sort_values(
        ["mean_precision", "worst_stage", "mean_coverage", "minimum_diversity"],
        ascending=[False, False, False, False],
        kind="stable",
    )
    return [
        {key: float(row[key]) for key in [
            "risk_weight", "evidence_weight", "need_weight", "certainty_weight", "workload_weight"
        ]}
        for _, row in ranking.head(count).iterrows()
    ]


def _select_from_training(
    frame: pd.DataFrame,
    protocol: dict[str, Any],
    *,
    salt: str,
) -> tuple[dict[str, Any], pd.DataFrame]:
    inner_count = int(protocol["evaluation"]["inner_group_folds"])
    working = frame.copy()
    working["inner_fold"] = working["base_record_id"].astype(str).map(
        lambda value: _stable_inner_fold(value, salt, inner_count)
    )
    fold_objects = []
    for inner_fold in range(inner_count):
        train = working[working["inner_fold"] != inner_fold].copy()
        validation = working[working["inner_fold"] == inner_fold].copy()
        if train.empty or validation.empty:
            raise RuntimeError(f"empty inner partition {inner_fold}")
        fold_objects.append((_group_arrays(validation), _fit_scales(train)))

    all_weights = _weight_configs(protocol)
    shortlist_count = int(protocol.get("search_strategy", {}).get("weight_shortlist", 12))
    shortlisted = _shortlist_weights(fold_objects, all_weights, shortlist_count)
    base_options = _base_thresholds(protocol)
    confidence_options = _confidence_thresholds(protocol)
    accumulator: dict[str, dict[str, Any]] = {}

    for weight in shortlisted:
        for arrays, scales in fold_objects:
            scores = _score_matrix(_components(arrays, scales), weight)
            for base in base_options:
                top = _top_selection(arrays, scores, base)
                for confidence in confidence_options:
                    config = {**weight, **base, **confidence}
                    config_id = _config_id(config)
                    record = accumulator.setdefault(
                        config_id,
                        {"config_id": config_id, **config, "fold_metrics": []},
                    )
                    record["fold_metrics"].append(_metrics(arrays, top, confidence))

    rows = []
    rule = protocol["selection_rule"]
    for record in accumulator.values():
        metrics = record.pop("fold_metrics")
        row = {
            **record,
            "mean_precision_at_1": float(np.mean([m["precision_at_1"] for m in metrics])),
            "mean_actionable_coverage": float(np.mean([m["actionable_coverage"] for m in metrics])),
            "mean_selective_accuracy": float(np.mean([m["selective_accuracy"] for m in metrics])),
            "mean_macro_action_precision": float(np.mean([m["macro_action_precision"] for m in metrics])),
            "worst_stage_precision": float(min(m["worst_stage_precision"] for m in metrics)),
            "minimum_action_diversity": int(min(m["action_diversity"] for m in metrics)),
            "maximum_top_action_concentration": float(max(m["top_action_concentration"] for m in metrics)),
        }
        row["meets_target"] = bool(
            row["mean_precision_at_1"] >= float(rule["target_precision"])
            and row["mean_actionable_coverage"] >= float(rule["minimum_inner_coverage"])
            and row["minimum_action_diversity"] >= int(rule["minimum_action_diversity"])
            and row["maximum_top_action_concentration"] <= float(rule["maximum_top_action_concentration"])
        )
        rows.append(row)
    trials = pd.DataFrame(rows)
    eligible = trials[trials["meets_target"]].copy()
    if eligible.empty:
        eligible = trials[
            trials["mean_actionable_coverage"] >= float(rule["minimum_inner_coverage"])
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
    keys = [
        "risk_weight", "evidence_weight", "need_weight", "certainty_weight", "workload_weight",
        "minimum_risk_reduction", "maximum_uncertainty", "minimum_evidence",
        "minimum_top_margin", "minimum_top_score",
    ]
    selected: dict[str, Any] = {key: float(selected_row[key]) for key in keys}
    selected.update(
        {
            "config_id": str(selected_row["config_id"]),
            "inner_target_met": bool(selected_row["meets_target"]),
            "shortlisted_weight_count": len(shortlisted),
            "total_weight_count": len(all_weights),
        }
    )
    return selected, trials


def _evaluate_partition(
    train: pd.DataFrame,
    test: pd.DataFrame,
    selected: dict[str, Any],
) -> tuple[pd.DataFrame, dict[str, Any], dict[str, float]]:
    scales = _fit_scales(train)
    arrays = _group_arrays(test)
    weights = {key: float(selected[key]) for key in [
        "risk_weight", "evidence_weight", "need_weight", "certainty_weight", "workload_weight"
    ]}
    base = {key: float(selected[key]) for key in [
        "minimum_risk_reduction", "maximum_uncertainty", "minimum_evidence"
    ]}
    confidence = {key: float(selected[key]) for key in [
        "minimum_top_margin", "minimum_top_score"
    ]}
    top = _top_selection(arrays, _score_matrix(_components(arrays, scales), weights), base)
    metrics = _metrics(arrays, top, confidence)
    issued = (
        top.exists
        & (top.top_score >= confidence["minimum_top_score"])
        & (top.top_margin >= confidence["minimum_top_margin"])
    )
    selected_family = np.asarray(ACTION_FAMILIES, dtype=object)[top.top_index]
    selected_runtime = np.asarray(RUNTIME_ACTIONS, dtype=object)[top.top_index]
    predictions = arrays.metadata.copy()
    predictions["action_family"] = selected_family
    predictions["runtime_action_id"] = selected_runtime
    predictions["hybrid_score"] = top.top_score
    predictions["top_margin"] = top.top_margin
    predictions["issued"] = issued.astype(int)
    predictions["silver_positive"] = top.silver.astype(int)
    predictions["correct_top1"] = (issued & (top.silver == 1)).astype(int)
    predictions["group_has_positive"] = (arrays.silver.max(axis=1) > 0).astype(int)
    predictions["risk_reduction"] = top.risk_reduction
    predictions["risk_uncertainty"] = top.uncertainty
    predictions["evidence_strength"] = top.evidence
    predictions["workload_minutes"] = top.workload
    metrics["selected_config_id"] = str(selected["config_id"])
    return predictions, metrics, scales


def _baseline_metrics(frame: pd.DataFrame) -> dict[str, dict[str, Any]]:
    arrays = _group_arrays(frame)
    permissive = {
        "minimum_risk_reduction": -1e9,
        "maximum_uncertainty": 1.0,
        "minimum_evidence": 0.0,
    }
    confidence = {"minimum_top_margin": -1e9, "minimum_top_score": -1e9}
    results = {}
    for name, scores in {
        "risk_reduction_only": arrays.risk_reduction,
        "evidence_only": arrays.evidence,
        "lowest_workload": -arrays.workload,
    }.items():
        results[name] = _metrics(arrays, _top_selection(arrays, scores, permissive), confidence)
    return results


def main() -> None:
    protocol = yaml.safe_load(PROTOCOL_PATH.read_text(encoding="utf-8"))
    frame = pd.read_parquet(DATA)
    required = {
        "group_id", "base_record_id", "stage", "outer_fold", "course", "presentation",
        "action_family", "runtime_action_id", "risk_reduction", "risk_uncertainty",
        "evidence_strength", "deficit_score", "workload_minutes", "action_available",
        "prerequisite_status", "silver_positive",
    }
    missing = required - set(frame.columns)
    if missing:
        raise RuntimeError(f"hybrid-only candidate schema missing: {sorted(missing)}")

    evaluation_out = OUT / "evaluation"
    selection_out = OUT / "model_selection"
    evaluation_out.mkdir(parents=True, exist_ok=True)
    selection_out.mkdir(parents=True, exist_ok=True)
    oof_rows = []
    fold_metrics = []
    baseline_rows = []

    for outer_fold in protocol["evaluation"]["outer_folds"]:
        train = frame[frame["outer_fold"] != outer_fold].copy()
        test = frame[frame["outer_fold"] == outer_fold].copy()
        selected, trials = _select_from_training(train, protocol, salt=f"outer-{outer_fold}-{SEED}")
        trials.to_csv(selection_out / f"fold_{outer_fold}_trials.csv", index=False)
        _atomic_json(selection_out / f"fold_{outer_fold}_selected.json", selected)
        predictions, metrics, scales = _evaluate_partition(train, test, selected)
        predictions["outer_fold"] = int(outer_fold)
        predictions.to_parquet(evaluation_out / f"fold_{outer_fold}_predictions.parquet", index=False)
        _atomic_json(
            evaluation_out / f"fold_{outer_fold}_metrics.json",
            {"metrics": metrics, "scales": scales, "selected": selected},
        )
        oof_rows.append(predictions)
        fold_metrics.append({"outer_fold": int(outer_fold), **metrics})
        for method, metrics_row in _baseline_metrics(test).items():
            baseline_rows.append({"outer_fold": int(outer_fold), "method": method, **metrics_row})

    oof = pd.concat(oof_rows, ignore_index=True)
    oof.to_parquet(evaluation_out / "OOF_PREDICTIONS.parquet", index=False)
    issued = oof[oof["issued"] == 1]
    positive_groups = int(oof["group_has_positive"].sum())
    overall = {
        "precision_at_1": float(issued["silver_positive"].mean()) if len(issued) else 0.0,
        "actionable_coverage": float(
            ((oof["issued"] == 1) & (oof["group_has_positive"] == 1)).sum() / positive_groups
        ) if positive_groups else 0.0,
        "issued_groups": int(len(issued)),
        "total_groups": int(len(oof)),
        "positive_groups": positive_groups,
        "action_diversity": int(issued["action_family"].nunique()) if len(issued) else 0,
        "top_action_concentration": float(issued["action_family"].value_counts(normalize=True).max()) if len(issued) else 1.0,
    }
    pd.DataFrame(fold_metrics).to_csv(evaluation_out / "FOLD_METRICS.csv", index=False)
    pd.DataFrame(baseline_rows).to_csv(evaluation_out / "BASELINE_METRICS.csv", index=False)

    final_selected, final_trials = _select_from_training(frame, protocol, salt=f"final-runtime-{SEED}")
    final_trials.to_csv(selection_out / "final_runtime_trials.csv", index=False)
    _atomic_json(
        OUT / "HYBRID_ONLY_SELECTED_CONFIG.json",
        {
            "status": "SELECTED_NOT_RELEASED",
            "config": final_selected,
            "normalization_scales": _fit_scales(frame),
            "additional_learned_model": False,
            "silver_labels_used_at_runtime": False,
            "claim_boundary": protocol["claim_boundary"],
            "execution_script": "tune_and_evaluate_fast.py",
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
            "runtime_filtering_order_matched": True,
        },
    )
    print(json.dumps(overall, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
