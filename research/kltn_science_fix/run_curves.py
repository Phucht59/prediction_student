"""P1: PR/ROC/confusion/reliability from OOF + labels. Illustrative, fold-median t."""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import PrecisionRecallDisplay, RocCurveDisplay, confusion_matrix

from .paths import FIG, RAW, REP, ROOT, ensure

OOF = ROOT / "artifacts" / "recommend_hybrid" / "v3" / "data" / "c0_oof_predictions.parquet"


def main() -> None:
    ensure()
    oof = pd.read_parquet(OOF)
    info = pd.read_csv(RAW / "studentInfo.csv")
    info["id_student"] = info.id_student.astype(str)
    oof["id_student"] = oof.id_student.astype(str)
    m = oof.merge(info, on=["id_student", "code_module", "code_presentation"], how="left")
    m["y"] = m.final_result.isin(["Fail", "Withdrawn"]).astype(int)
    stages = ["20pct", "35pct", "50pct", "75pct"]
    fig, axes = plt.subplots(2, 2, figsize=(9, 8))
    for ax, st in zip(axes.ravel(), stages):
        sub = m[m.stage_or_endpoint == st]
        PrecisionRecallDisplay.from_predictions(sub.y, sub.risk_probability, ax=ax, name=st)
        ax.set_title(f"PR {st}")
    fig.tight_layout()
    fig.savefig(FIG / "pr_curves_oulad_oof.png", dpi=150)
    plt.close(fig)
    fig, axes = plt.subplots(2, 2, figsize=(9, 8))
    for ax, st in zip(axes.ravel(), stages):
        sub = m[m.stage_or_endpoint == st]
        RocCurveDisplay.from_predictions(sub.y, sub.risk_probability, ax=ax, name=st)
        ax.set_title(f"ROC {st}")
    fig.tight_layout()
    fig.savefig(FIG / "roc_curves_oulad_oof.png", dpi=150)
    plt.close(fig)
    fig, axes = plt.subplots(2, 2, figsize=(9, 8))
    for ax, st in zip(axes.ravel(), stages):
        sub = m[m.stage_or_endpoint == st]
        t = float(sub.prediction_threshold.median())
        cm = confusion_matrix(sub.y, (sub.risk_probability >= t).astype(int), labels=[0, 1])
        ax.imshow(cm, cmap="Blues")
        ax.set_title(f"CM {st} t={t:.2f}")
        for (i, j), val in np.ndenumerate(cm):
            ax.text(j, i, str(val), ha="center", va="center")
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xlabel("pred")
        ax.set_ylabel("true")
    fig.tight_layout()
    fig.savefig(FIG / "confusion_oulad_oof.png", dpi=150)
    plt.close(fig)
    fig, axes = plt.subplots(2, 2, figsize=(9, 8))
    for ax, st in zip(axes.ravel(), stages):
        sub = m[m.stage_or_endpoint == st]
        frac, meanp = calibration_curve(sub.y, sub.risk_probability, n_bins=12, strategy="uniform")
        ax.plot(meanp, frac, marker="o")
        ax.plot([0, 1], [0, 1], "--", color="gray")
        ax.set_title(f"Reliability {st}")
        ax.set_xlabel("mean p")
        ax.set_ylabel("fraction positive")
    fig.tight_layout()
    fig.savefig(FIG / "reliability_oulad_oof.png", dpi=150)
    plt.close(fig)
    (REP / "CURVES_NOTE.md").write_text(
        "\n".join(
            [
                "# PR / ROC / confusion / reliability",
                "",
                "OOF serving (không chứa cột target) được join `final_result` để vẽ. Ngưỡng = median `prediction_threshold` theo mốc — minh họa vận hành, không phải confusion từ file không nhãn.",
                "Hình: `pr_curves_oulad_oof.png`, `roc_curves_oulad_oof.png`, `confusion_oulad_oof.png`, `reliability_oulad_oof.png`.",
                "",
            ]
        ),
        encoding="utf-8",
    )
    print("wrote curves")


if __name__ == "__main__":
    main()
