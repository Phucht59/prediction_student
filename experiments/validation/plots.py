"""Write calibration and confusion artifacts. No probability rescaling."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def save_reliability(path: Path, bins: list[dict], title: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not bins:
        return
    xs = [row["mean_p"] for row in bins]
    ys = [row["mean_y"] for row in bins]
    ns = [row["n"] for row in bins]
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot([0, 1], [0, 1], "--", color="gray", label="perfect")
    ax.scatter(xs, ys, s=[max(12, n / 5) for n in ns], color="C0")
    ax.plot(xs, ys, color="C0")
    ax.set_xlabel("Mean predicted P(risk)")
    ax.set_ylabel("Observed risk rate")
    ax.set_title(title)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def save_confusion(path: Path, tp: int, fp: int, tn: int, fn: int, title: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    mat = np.array([[tn, fp], [fn, tp]], dtype=float)
    fig, ax = plt.subplots(figsize=(4.2, 3.6))
    im = ax.imshow(mat, cmap="Blues")
    ax.set_xticks([0, 1], ["Pred 0", "Pred 1"])
    ax.set_yticks([0, 1], ["True 0", "True 1"])
    for (i, j), val in np.ndenumerate(mat):
        ax.text(j, i, int(val), ha="center", va="center")
    ax.set_title(title)
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
