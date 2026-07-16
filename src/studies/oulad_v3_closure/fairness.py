from __future__ import annotations

import hashlib
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, f1_score, precision_score, recall_score

from src.studies.oulad_v2.metrics import expected_calibration_error, prediction_frame_metrics


DECLARED_SEEDS = (42, 2026, 3407)


def sha256(path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_seed_coverage(frame: pd.DataFrame, candidate_id: str) -> dict[str, object]:
    candidate = frame.loc[frame["candidate_id"] == candidate_id].copy()
    if tuple(sorted(candidate["seed"].unique())) != DECLARED_SEEDS:
        raise RuntimeError(f"{candidate_id}: declared seed coverage failed")
    if candidate.duplicated(["record_id", "seed"]).any():
        raise RuntimeError(f"{candidate_id}: duplicate record/seed")
    identities = []
    for seed in DECLARED_SEEDS:
        part = candidate.loc[candidate["seed"] == seed].sort_values("record_id")
        identities.append((tuple(part["record_id"]), tuple(part["outer_fold"]), tuple(part["target_at_risk"])))
    if identities[1:] != identities[:-1]:
        raise RuntimeError(f"{candidate_id}: record/fold/label mismatch between seeds")
    if not np.isfinite(candidate["probability"]).all() or not candidate["probability"].between(0, 1).all():
        raise RuntimeError(f"{candidate_id}: invalid probability")
    return {
        "candidate_id": candidate_id,
        "seeds": list(DECLARED_SEEDS),
        "records_per_seed": int(len(candidate) / 3),
        "unique_records": int(candidate["record_id"].nunique()),
        "target_prevalence": float(candidate.loc[candidate.seed == 42, "target_at_risk"].mean()),
        "status": "PASS",
    }


def ensemble_outer_predictions(source: pd.DataFrame, source_candidate: str, ensemble_candidate: str,
                               thresholds: pd.DataFrame) -> pd.DataFrame:
    validate_seed_coverage(source, source_candidate)
    candidate = source.loc[source.candidate_id == source_candidate].copy()
    keys = ["forecast_id", "outer_fold", "record_id", "code_module", "code_presentation", "id_student", "target_at_risk"]
    result = candidate.groupby(keys, as_index=False).agg(probability=("probability", "mean"))
    result = result.merge(thresholds.loc[thresholds.candidate_id == ensemble_candidate,
        ["outer_fold", "macro_threshold", "operational_threshold", "operational_feasible"]], on="outer_fold", validate="many_to_one")
    result.insert(0, "candidate_id", ensemble_candidate)
    result.insert(2, "scope", "fair_ensemble_closure")
    result.insert(4, "seed", np.nan)
    result["prediction_contract"] = "probability_ensemble"
    result["prediction_variant"] = "ensemble_42_2026_3407"
    result["predicted_label"] = (result.probability >= result.macro_threshold).astype(int)
    result["operational_prediction"] = (result.probability >= result.operational_threshold).astype(int)
    return result


def metrics_with_modules(frame: pd.DataFrame, minimum_records=60, minimum_positive=10, minimum_negative=10) -> dict[str, object]:
    metrics = prediction_frame_metrics(frame)
    module_rows = []
    for module, part in frame.groupby("code_module"):
        positive = int(part.target_at_risk.sum()); negative = int(len(part) - positive)
        if len(part) >= minimum_records and positive >= minimum_positive and negative >= minimum_negative:
            item = prediction_frame_metrics(part)
            module_rows.append((module, item["macro_f1"], item["at_risk_recall"]))
    metrics["worst_eligible_module_macro_f1"] = min(row[1] for row in module_rows)
    metrics["worst_eligible_module_recall"] = min(row[2] for row in module_rows)
    metrics["eligible_module_count"] = len(module_rows)
    return metrics


def _macro_f1(weight: np.ndarray, y: np.ndarray, prediction: np.ndarray) -> np.ndarray:
    tp = weight @ ((y == 1) & (prediction == 1)); fp = weight @ ((y == 0) & (prediction == 1))
    fn = weight @ ((y == 1) & (prediction == 0)); tn = weight @ ((y == 0) & (prediction == 0))
    f1_pos = np.divide(2 * tp, 2 * tp + fp + fn, out=np.zeros_like(tp, dtype=float), where=(2 * tp + fp + fn) > 0)
    f1_neg = np.divide(2 * tn, 2 * tn + fp + fn, out=np.zeros_like(tn, dtype=float), where=(2 * tn + fp + fn) > 0)
    return (f1_pos + f1_neg) / 2


def _recall(weight: np.ndarray, y: np.ndarray, prediction: np.ndarray) -> np.ndarray:
    tp = weight @ ((y == 1) & (prediction == 1)); fn = weight @ ((y == 1) & (prediction == 0))
    return np.divide(tp, tp + fn, out=np.zeros_like(tp, dtype=float), where=(tp + fn) > 0)


def _precision(weight: np.ndarray, y: np.ndarray, prediction: np.ndarray) -> np.ndarray:
    tp = weight @ ((y == 1) & (prediction == 1)); fp = weight @ ((y == 0) & (prediction == 1))
    return np.divide(tp, tp + fp, out=np.zeros_like(tp, dtype=float), where=(tp + fp) > 0)


def _weighted_ap(weight: np.ndarray, y: np.ndarray, probability: np.ndarray) -> np.ndarray:
    # Exact for unique scores; score ties are kept adjacent with deterministic stable ordering.
    order = np.argsort(-probability, kind="mergesort"); labels = y[order]; ordered = weight[:, order]
    tp = np.cumsum(ordered * labels, axis=1); predicted = np.cumsum(ordered, axis=1)
    precision = np.divide(tp, predicted, out=np.zeros_like(tp, dtype=float), where=predicted > 0)
    positive_weight = ordered * labels; total_positive = positive_weight.sum(axis=1)
    return np.divide((precision * positive_weight).sum(axis=1), total_positive,
                     out=np.zeros_like(total_positive, dtype=float), where=total_positive > 0)


def grouped_bootstrap_pair(left: pd.DataFrame, right: pd.DataFrame, *, resamples: int, seed: int,
                           superiority_margin: float = .005, batch_size: int = 100) -> list[dict[str, object]]:
    keys = ["record_id", "id_student", "target_at_risk", "code_module"]
    a = left[keys + ["probability", "predicted_label", "operational_prediction"]].rename(columns={
        "probability": "probability_left", "predicted_label": "prediction_left", "operational_prediction": "operational_left"})
    b = right[keys + ["probability", "predicted_label", "operational_prediction"]].rename(columns={
        "probability": "probability_right", "predicted_label": "prediction_right", "operational_prediction": "operational_right"})
    merged = a.merge(b, on=keys, validate="one_to_one").reset_index(drop=True)
    if len(merged) != len(left) or len(left) != len(right):
        raise RuntimeError("Paired bootstrap coverage mismatch")
    group_codes, groups = pd.factorize(merged.id_student, sort=True); n_groups = len(groups)
    y = merged.target_at_risk.to_numpy(dtype=int); pl = merged.prediction_left.to_numpy(dtype=int); pr = merged.prediction_right.to_numpy(dtype=int)
    ol = merged.operational_left.to_numpy(dtype=int); oright = merged.operational_right.to_numpy(dtype=int)
    prob_l = merged.probability_left.to_numpy(dtype=float); prob_r = merged.probability_right.to_numpy(dtype=float)
    modules = sorted(merged.code_module.unique()); rng = np.random.default_rng(seed)
    values = {name: [] for name in ["macro_f1", "pr_auc", "at_risk_recall", "worst_module_macro_f1", "operational_recall_at_precision_gte_0.75"]}
    feasibility_left = []; feasibility_right = []
    for start in range(0, resamples, batch_size):
        size = min(batch_size, resamples - start)
        group_weight = rng.multinomial(n_groups, np.full(n_groups, 1 / n_groups), size=size).astype(np.float32)
        weight = group_weight[:, group_codes]
        values["macro_f1"].extend((_macro_f1(weight, y, pl) - _macro_f1(weight, y, pr)).tolist())
        values["at_risk_recall"].extend((_recall(weight, y, pl) - _recall(weight, y, pr)).tolist())
        values["pr_auc"].extend((_weighted_ap(weight, y, prob_l) - _weighted_ap(weight, y, prob_r)).tolist())
        module_left=[]; module_right=[]
        for module in modules:
            selected=(merged.code_module.to_numpy()==module); module_weight=weight[:,selected]
            module_left.append(_macro_f1(module_weight,y[selected],pl[selected])); module_right.append(_macro_f1(module_weight,y[selected],pr[selected]))
        values["worst_module_macro_f1"].extend((np.min(np.stack(module_left),axis=0)-np.min(np.stack(module_right),axis=0)).tolist())
        precision_l=_precision(weight,y,ol); precision_r=_precision(weight,y,oright)
        feasible_l=precision_l>=.75; feasible_r=precision_r>=.75; both=feasible_l&feasible_r
        op_delta=np.full(size,np.nan); op_delta[both]=_recall(weight[both],y,ol)[0:both.sum()]-_recall(weight[both],y,oright)[0:both.sum()]
        values["operational_recall_at_precision_gte_0.75"].extend(op_delta.tolist()); feasibility_left.extend(feasible_l.tolist()); feasibility_right.extend(feasible_r.tolist())
    rows=[]
    for metric,raw in values.items():
        vector=np.asarray(raw,dtype=float); finite=vector[np.isfinite(vector)]
        rows.append({"left_candidate":left.candidate_id.iloc[0],"right_candidate":right.candidate_id.iloc[0],
            "prediction_contract":"fair_probability_ensemble_or_registered_deterministic","metric":metric,
            "mean_delta":float(np.mean(finite)),"median_delta":float(np.median(finite)),"lower_95":float(np.quantile(finite,.025)),
            "upper_95":float(np.quantile(finite,.975)),"probability_delta_gt_zero":float(np.mean(finite>0)),
            "probability_delta_gte_superiority_margin":float(np.mean(finite>=superiority_margin)),"students":n_groups,"records":len(merged),
            "resamples":resamples,"operational_pair_feasibility_rate":float(len(finite)/resamples) if metric.startswith("operational") else None,
            "left_operational_feasibility_rate":float(np.mean(feasibility_left)) if metric.startswith("operational") else None,
            "right_operational_feasibility_rate":float(np.mean(feasibility_right)) if metric.startswith("operational") else None})
    return rows
