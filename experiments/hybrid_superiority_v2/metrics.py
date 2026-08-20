"""Primary metric is Average Precision (AP), not trapezoidal PR-AUC."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    auc as trapz_auc,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)

from .protocol import material_margin, normalized_margin, scalar_objective, warm_for

THRESHOLD_GRID = tuple(round(value / 100, 2) for value in range(5, 96))


def expected_calibration_error(target, score, n_bins: int = 15) -> float:
    target = np.asarray(target, dtype=float)
    score = np.asarray(score, dtype=float)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        right = bins[i + 1] if i < n_bins - 1 else bins[i + 1] + 1e-12
        mask = (score >= bins[i]) & (score < right)
        if mask.any():
            ece += float(mask.mean() * abs(score[mask].mean() - target[mask].mean()))
    return float(ece)


def recall_at_alert_budget(target, score, budget: float) -> float:
    target = np.asarray(target, dtype=int)
    score = np.asarray(score, dtype=float)
    positives = int(target.sum())
    if positives == 0:
        return 0.0
    k = max(1, int(np.ceil(budget * len(target))))
    order = np.argsort(-score)[:k]
    return float(target[order].sum() / positives)


def precision_at_recall(target, score, min_recall: float = 0.80) -> float | None:
    target = np.asarray(target, dtype=int)
    score = np.asarray(score, dtype=float)
    precision, recall, _ = precision_recall_curve(target, score)
    ok = recall >= min_recall
    if not ok.any():
        return None
    return float(precision[ok].max())


def trapezoidal_pr_auc(target, score) -> float:
    precision, recall, _ = precision_recall_curve(target, score)
    return float(trapz_auc(recall, precision))


def binary_metrics(target, score, *, threshold: float = 0.5) -> dict[str, float]:
    target = np.asarray(target, dtype=int)
    score = np.asarray(score, dtype=float)
    if target.size == 0:
        raise ValueError("empty target")
    if not np.isfinite(score).all():
        raise ValueError("nonfinite scores")
    if len(np.unique(target)) < 2:
        raise ValueError("single-class partition")
    pred = (score >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(target, pred, labels=[0, 1]).ravel()
    clipped = np.clip(score, 1e-6, 1.0 - 1e-6)
    return {
        "ap": float(average_precision_score(target, score)),
        "pr_auc_trapezoid": trapezoidal_pr_auc(target, score),
        "roc_auc": float(roc_auc_score(target, score)),
        "risk_precision": float(precision_score(target, pred, pos_label=1, zero_division=0)),
        "risk_recall": float(recall_score(target, pred, pos_label=1, zero_division=0)),
        "risk_f1": float(f1_score(target, pred, pos_label=1, zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(target, pred)),
        "accuracy": float(accuracy_score(target, pred)),
        "ece": expected_calibration_error(target, score),
        "brier": float(brier_score_loss(target, score)),
        "log_loss": float(log_loss(target, clipped, labels=[0, 1])),
        "recall_at_10": recall_at_alert_budget(target, score, 0.10),
        "recall_at_20": recall_at_alert_budget(target, score, 0.20),
        "threshold": float(threshold),
        "tp": int(tp),
        "fp": int(fp),
        "tn": int(tn),
        "fn": int(fn),
        "n": int(len(target)),
        "prevalence": float(target.mean()),
    }


def select_stop_threshold(target, score) -> float:
    ranked = []
    for threshold in THRESHOLD_GRID:
        metrics = binary_metrics(target, score, threshold=threshold)
        ranked.append((metrics["risk_f1"], metrics["risk_recall"], -abs(threshold - 0.5), threshold))
    return float(max(ranked)[-1])


def stage_deltas(hybrid_ap: dict[str, float], baseline_ap: dict[str, float]) -> dict[str, dict[str, float]]:
    out = {}
    for stage, h in hybrid_ap.items():
        b = float(baseline_ap[stage])
        delta = float(h) - b
        out[stage] = {
            "ap_hybrid": float(h),
            "ap_baseline": b,
            "delta_ap": delta,
            "material_margin": material_margin(b),
            "normalized_margin": normalized_margin(delta, b),
        }
    return out


def selection_objective(hybrid_ap: dict[str, float], baseline_ap: dict[str, float], domain: str, n_params: int, fold_seed_std: float = 0.0) -> dict[str, float]:
    warm = warm_for(domain)
    rows = stage_deltas({s: hybrid_ap[s] for s in warm}, {s: baseline_ap[s] for s in warm})
    r = [rows[s]["normalized_margin"] for s in warm]
    n_fail = int(sum(1 for s in warm if rows[s]["delta_ap"] <= 0))
    j = scalar_objective(r, fold_seed_std, n_params)
    return {
        "n_warm_nonpositive": n_fail,
        "min_normalized_margin": float(min(r)),
        "mean_clipped_normalized_margin": float(np.clip(r, -2, 2).mean()),
        "J": j,
        "n_params": int(n_params),
    }
