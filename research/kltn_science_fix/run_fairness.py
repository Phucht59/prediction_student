"""P0.4 AP by gender / imd_band / disability / code_module. Descriptive only."""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

from .paths import FIG, RAW, REP, ROOT, ensure

OOF = ROOT / "artifacts" / "recommend_hybrid" / "v3" / "data" / "c0_oof_predictions.parquet"


def ap_safe(y, p) -> float:
    y = np.asarray(y, int)
    p = np.asarray(p, float)
    if y.size < 30 or len(np.unique(y)) < 2:
        return float("nan")
    return float(average_precision_score(y, p))


def main() -> None:
    ensure()
    oof = pd.read_parquet(OOF)
    info = pd.read_csv(RAW / "studentInfo.csv")
    info["id_student"] = info.id_student.astype(str)
    oof["id_student"] = oof.id_student.astype(str)
    m = oof.merge(info, on=["id_student", "code_module", "code_presentation"], how="left")
    m["y"] = m.final_result.isin(["Fail", "Withdrawn"]).astype(int)
    rows = []
    for col in ("gender", "imd_band", "disability", "code_module"):
        for stage, sub in m.groupby("stage_or_endpoint"):
            t = float(sub.prediction_threshold.median())
            for g, gg in sub.groupby(col):
                pred = (gg.risk_probability >= t).astype(int)
                fn = int(((gg.y == 1) & (pred == 0)).sum())
                n_pos = int(gg.y.sum())
                rows.append(
                    {
                        "attribute": col,
                        "group": str(g),
                        "stage": stage,
                        "n": int(len(gg)),
                        "n_pos": n_pos,
                        "prevalence": float(gg.y.mean()),
                        "ap": ap_safe(gg.y, gg.risk_probability),
                        "fn_rate": fn / n_pos if n_pos else float("nan"),
                    }
                )
    frame = pd.DataFrame(rows)
    frame.to_csv(REP / "fairness_by_group.csv", index=False)
    # flag groups with AP much below stage overall
    flags = []
    for stage in frame.stage.unique():
        overall = ap_safe(m.loc[m.stage_or_endpoint == stage, "y"], m.loc[m.stage_or_endpoint == stage, "risk_probability"])
        part = frame[(frame.stage == stage) & frame.ap.notna() & (frame.n >= 200)]
        for r in part.itertuples():
            if r.ap < overall - 0.05:
                flags.append({**r._asdict(), "stage_ap": overall, "gap": r.ap - overall})
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    for ax, col in zip(axes.ravel(), ("gender", "imd_band", "disability", "code_module")):
        sub = frame[(frame.attribute == col) & (frame.stage == "35pct")]
        ax.bar(sub.group.astype(str), sub.ap, color="#1f4e79")
        ax.set_title(f"OULAD 35% AP by {col}")
        ax.set_ylim(0.5, 1.0)
        ax.tick_params(axis="x", rotation=45, labelsize=8)
    fig.tight_layout()
    fig.savefig(FIG / "fairness_ap_by_group_35pct.png", dpi=150)
    plt.close(fig)
    lines = [
        "# FAIRNESS_BY_GROUP",
        "",
        "Phân tích **mô tả**, không phải can thiệp sửa bias. OOF serving join studentInfo.",
        "",
        f"CSV: `{ (REP / 'fairness_by_group.csv').as_posix() }`.",
        f"Hình 35%: `{ (FIG / 'fairness_ap_by_group_35pct.png').as_posix() }`.",
        "",
        "## Nhóm AP thấp hơn overall ≥ 0.05 (n≥200)",
        "",
    ]
    if flags:
        lines.append("| attr | group | stage | n | AP | stage AP | Δ | FN rate |")
        lines.append("|---|---|---|---:|---:|---:|---:|---:|")
        for f in flags:
            lines.append(
                f"| {f['attribute']} | {f['group']} | {f['stage']} | {f['n']} | {f['ap']:.3f} | {f['stage_ap']:.3f} | {f['gap']:+.3f} | {f['fn_rate']:.3f} |"
            )
    else:
        lines.append("Không nhóm nào (n≥200) thấp hơn overall 0.05 trên các mốc OOF.")
    lines.append("")
    (REP / "FAIRNESS_BY_GROUP.md").write_text("\n".join(lines), encoding="utf-8")
    print("wrote", REP / "FAIRNESS_BY_GROUP.md")


if __name__ == "__main__":
    main()
