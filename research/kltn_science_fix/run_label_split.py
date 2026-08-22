"""P0.3 Fail vs Withdrawn on serving OOF (seed 42, 3 folds)."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

from .metrics import binary_metrics
from .paths import RAW, REP, ROOT, ensure

OOF = ROOT / "artifacts" / "recommend_hybrid" / "v3" / "data" / "c0_oof_predictions.parquet"


def ap_safe(y, p) -> float:
    y = np.asarray(y, int)
    p = np.asarray(p, float)
    if y.size == 0 or len(np.unique(y)) < 2:
        return float("nan")
    return float(average_precision_score(y, p))


def main() -> None:
    ensure()
    oof = pd.read_parquet(OOF)
    info = pd.read_csv(RAW / "studentInfo.csv")
    info["id_student"] = info.id_student.astype(str)
    oof["id_student"] = oof.id_student.astype(str)
    merged = oof.merge(info, on=["id_student", "code_module", "code_presentation"], how="left")
    merged["final_result"] = merged.final_result.astype(str)
    merged["y_combo"] = merged.final_result.isin(["Fail", "Withdrawn"]).astype(int)
    rows = []
    for stage, sub in merged.groupby("stage_or_endpoint"):
        fail = sub.final_result == "Fail"
        wd = sub.final_result == "Withdrawn"
        rest_ok = sub.final_result.isin(["Pass", "Distinction"])
        t = float(sub.prediction_threshold.median())
        rows.append(
            {
                "stage": stage,
                "n": int(len(sub)),
                "n_fail": int(fail.sum()),
                "n_withdrawn": int(wd.sum()),
                "n_passdist": int(rest_ok.sum()),
                "ap_combo": ap_safe(sub.y_combo, sub.risk_probability),
                "ap_fail_vs_pass": ap_safe(fail[fail | rest_ok], sub.loc[fail | rest_ok, "risk_probability"]),
                "ap_withdrawn_vs_pass": ap_safe(wd[wd | rest_ok], sub.loc[wd | rest_ok, "risk_probability"]),
                "threshold_median": t,
                **{f"combo_{k}": v for k, v in binary_metrics(sub.y_combo, sub.risk_probability, threshold=t).items() if k in {"precision", "recall", "f1"}},
            }
        )
    frame = pd.DataFrame(rows).sort_values("stage")
    frame.to_csv(REP.parent / ".." / "research" / "hybrid_superiority_v2" / "label_split.csv", index=False)
    out_csv = REP / "label_split.csv"
    frame.to_csv(out_csv, index=False)
    lines = [
        "# LABEL_SPLIT_ANALYSIS",
        "",
        "OOF serving 66 685 dòng (20/35/50/75, 3 inner fold, seed 42) join `studentInfo.final_result`.",
        "Không có mốc 100% trong OOF này.",
        "",
        "| stage | n | Fail | Withdrawn | Pass/Dist | AP gộp | AP Fail vs Pass | AP Withdrawn vs Pass |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in frame.itertuples():
        lines.append(
            f"| {r.stage} | {r.n} | {r.n_fail} | {r.n_withdrawn} | {r.n_passdist} | {r.ap_combo:.4f} | {r.ap_fail_vs_pass:.4f} | {r.ap_withdrawn_vs_pass:.4f} |"
        )
    lines += [
        "",
        "AP Fail vs Pass loại Withdrawn khỏi đánh giá; AP Withdrawn vs Pass loại Fail.",
        "Serving AP 100% = 0.9204 (bảng khóa) **không** nằm file OOF này. 100% còn ~94 Withdrawn sau lọc cutoff — không dùng làm bằng chứng cảnh báo sớm.",
        f"CSV: `{out_csv.as_posix()}`.",
        "",
    ]
    (REP / "LABEL_SPLIT_ANALYSIS.md").write_text("\n".join(lines), encoding="utf-8")
    print("wrote", REP / "LABEL_SPLIT_ANALYSIS.md")


if __name__ == "__main__":
    main()
