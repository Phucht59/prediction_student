from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

GRID = tuple(round(v / 100, 2) for v in range(5, 96))


def ece(y, p, n_bins: int = 15) -> float:
    y = np.asarray(y, float)
    p = np.asarray(p, float)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    out = 0.0
    for i in range(n_bins):
        right = bins[i + 1] + (1e-12 if i == n_bins - 1 else 0.0)
        m = (p >= bins[i]) & (p < right)
        if m.any():
            out += float(m.mean() * abs(p[m].mean() - y[m].mean()))
    return float(out)


def binary_metrics(y, p, *, threshold: float = 0.5) -> dict:
    y = np.asarray(y, int)
    p = np.asarray(p, float)
    if y.size == 0 or len(np.unique(y)) < 2:
        return {
            "ap": float("nan"),
            "roc_auc": float("nan"),
            "precision": float("nan"),
            "recall": float("nan"),
            "f1": float("nan"),
            "accuracy": float("nan"),
            "ece": float("nan"),
            "brier": float("nan"),
            "threshold": float(threshold),
            "tp": 0,
            "fp": 0,
            "tn": 0,
            "fn": 0,
            "n": int(y.size),
            "n_pos": int(y.sum()) if y.size else 0,
        }
    pred = (p >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {
        "ap": float(average_precision_score(y, p)),
        "roc_auc": float(roc_auc_score(y, p)),
        "precision": float(precision_score(y, pred, pos_label=1, zero_division=0)),
        "recall": float(recall_score(y, pred, pos_label=1, zero_division=0)),
        "f1": float(f1_score(y, pred, pos_label=1, zero_division=0)),
        "accuracy": float(accuracy_score(y, pred)),
        "ece": ece(y, p),
        "brier": float(brier_score_loss(y, p)),
        "threshold": float(threshold),
        "tp": int(tp),
        "fp": int(fp),
        "tn": int(tn),
        "fn": int(fn),
        "n": int(y.size),
        "n_pos": int(y.sum()),
    }


def select_threshold(y, p) -> float:
    ranked = []
    for t in GRID:
        m = binary_metrics(y, p, threshold=t)
        ranked.append((m["f1"], m["recall"], -abs(t - 0.5), t))
    return float(max(ranked)[-1])
