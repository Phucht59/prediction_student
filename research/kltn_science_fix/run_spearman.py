"""P1: Spearman on FIT-only vs full-n exploratory."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from .data import copy_locked_splits, inner_partitions
from .paths import RAW, REP, SPLIT, ensure
from src.prediction.data.uci import UCI_NUMERIC_CONTEXT, build_uci_combined


def main() -> None:
    ensure()
    copy_locked_splits()
    frame, _ = build_uci_combined(RAW / "student-mat.csv", RAW / "student-por.csv")
    frame = frame.copy()
    frame["record_id"] = frame.record_id.astype(str)
    frame["group_id"] = frame.global_student_group.astype(str)
    y = (frame.G3 < 10).astype(int)
    cols = ["G1", "G2", "G3", "failures", "age", "absences", *UCI_NUMERIC_CONTEXT]
    cols = list(dict.fromkeys([c for c in cols if c in frame.columns]))
    full_rows = []
    for c in cols:
        rho, p = spearmanr(frame[c], y)
        full_rows.append({"split": "all_n1044_exploratory", "column": c, "rho": float(rho), "p": float(p)})
    fit_rhos = {c: [] for c in cols}
    for fold in (0, 1, 2):
        fit, stop, valid = inner_partitions("uci", frame, fold)
        sub = frame[frame.record_id.isin(fit)]
        yf = (sub.G3 < 10).astype(int)
        for c in cols:
            rho, p = spearmanr(sub[c], yf)
            fit_rhos[c].append(float(rho))
            full_rows.append({"split": f"FIT_fold{fold}", "column": c, "rho": float(rho), "p": float(p)})
    for c in cols:
        full_rows.append(
            {
                "split": "FIT_mean_3fold",
                "column": c,
                "rho": float(np.mean(fit_rhos[c])),
                "p": float("nan"),
            }
        )
    out = pd.DataFrame(full_rows)
    out.to_csv(REP / "spearman_fit_vs_full.csv", index=False)
    pivot = (
        out[out.split.isin(["all_n1044_exploratory", "FIT_mean_3fold"])]
        .pivot(index="column", columns="split", values="rho")
        .round(3)
    )
    table = ["| column | all_n1044_exploratory | FIT_mean_3fold |", "|---|---:|---:|"]
    for col, row in pivot.iterrows():
        table.append(
            f"| {col} | {row.get('all_n1044_exploratory', float('nan')):.3f} | {row.get('FIT_mean_3fold', float('nan')):.3f} |"
        )
    (REP / "SPEARMAN_FIT.md").write_text(
        "\n".join(
            [
                "# Spearman FIT-only vs full n",
                "",
                "Bản `all_n1044_exploratory` = EDA cũ (nhìn cả VALID). Bản `FIT_mean_3fold` chỉ trên FIT, outer fold 0 firewall.",
                f"CSV: `{(REP / 'spearman_fit_vs_full.csv').as_posix()}`.",
                "",
                *table,
                "",
            ]
        ),
        encoding="utf-8",
    )
    print("wrote spearman")


if __name__ == "__main__":
    main()
