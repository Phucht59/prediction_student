"""CPU-only feasibility of OULAD multiclass. Does not change the locked binary task."""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
)

from src.prediction.data.oulad import load_oulad_static_tables
from src.prediction.data.oulad_features import (
    OULAD_CATEGORICAL_CONTEXT,
    OULAD_NUMERIC_CONTEXT,
    STATE_FRACTIONS,
    eligible_oulad,
)
from src.prediction.data.preprocessing import ContextPreprocessor
from research.kltn_science_fix.data import copy_locked_splits, inner_partitions
from research.kltn_science_fix.paths import RAW, REP, ensure

STAGES = ("20pct", "35pct", "50pct", "75pct", "100pct")
CLAIM = ("35pct", "50pct", "75pct")
SCHEMES = {
    "4class": ["Distinction", "Pass", "Fail", "Withdrawn"],
    "3class_passband": ["PassBand", "Fail", "Withdrawn"],
}


def _map(labels: pd.Series, scheme: str) -> pd.Series:
    if scheme == "4class":
        return labels.astype(str)
    mapped = labels.astype(str).replace({"Distinction": "PassBand", "Pass": "PassBand"})
    return mapped


def main() -> None:
    ensure()
    copy_locked_splits()
    _, _, base = load_oulad_static_tables(RAW)
    counts_rows = []
    for stage in STAGES:
        elig = eligible_oulad(base, STATE_FRACTIONS[stage])
        vc = elig.final_result.astype(str).value_counts()
        n = len(elig)
        row = {"stage": stage, "n": n}
        for name in ["Pass", "Distinction", "Fail", "Withdrawn"]:
            row[name] = int(vc.get(name, 0))
            row[f"{name}_pct"] = float(vc.get(name, 0) / n) if n else 0.0
        row["risk_binary"] = int(vc.get("Fail", 0) + vc.get("Withdrawn", 0))
        row["risk_binary_pct"] = row["risk_binary"] / n if n else 0.0
        counts_rows.append(row)
    counts = pd.DataFrame(counts_rows)
    counts.to_csv(REP / "multiclass_counts.csv", index=False)

    context = base[
        ["record_id", "group_id", "target", "final_result", *OULAD_NUMERIC_CONTEXT, *OULAD_CATEGORICAL_CONTEXT]
    ].copy()
    context["record_id"] = context.record_id.astype(str)
    context["group_id"] = context.group_id.astype(str)
    fit_ids, stop_ids, valid_ids = inner_partitions("oulad", context, 0)
    fit_set, valid_set = set(fit_ids), set(valid_ids)

    model_rows = []
    reports = {}
    for stage in CLAIM:
        elig = eligible_oulad(base, STATE_FRACTIONS[stage])
        elig["record_id"] = elig.record_id.astype(str)
        fit = elig[elig.record_id.isin(fit_set)].drop_duplicates("record_id")
        valid = elig[elig.record_id.isin(valid_set)].drop_duplicates("record_id")
        prep = ContextPreprocessor(list(OULAD_NUMERIC_CONTEXT), list(OULAD_CATEGORICAL_CONTEXT)).fit(fit)
        x_fit = prep.transform(fit)
        x_valid = prep.transform(valid)
        y_bin_v = valid.target.to_numpy()
        # binary LR reference on same static features
        bin_clf = LogisticRegression(max_iter=400, class_weight="balanced")
        bin_clf.fit(x_fit, fit.target.to_numpy())
        p_bin = bin_clf.predict_proba(x_valid)[:, 1]
        ap_bin = float(average_precision_score(y_bin_v, p_bin)) if len(np.unique(y_bin_v)) == 2 else float("nan")

        for scheme, classes in SCHEMES.items():
            y_fit = _map(fit.final_result, scheme)
            y_valid = _map(valid.final_result, scheme)
            present = [c for c in classes if c in set(y_fit.unique())]
            if len(present) < 2:
                continue
            clf = LogisticRegression(max_iter=400, class_weight="balanced")
            clf.fit(x_fit, y_fit)
            pred = clf.predict(x_valid)
            acc = float(accuracy_score(y_valid, pred))
            f1m = float(f1_score(y_valid, pred, average="macro", zero_division=0))
            f1w = float(f1_score(y_valid, pred, average="weighted", zero_division=0))
            per = f1_score(y_valid, pred, average=None, labels=present, zero_division=0)
            cm = confusion_matrix(y_valid, pred, labels=present)
            # Withdrawn support on VALID
            support = {c: int((y_valid == c).sum()) for c in present}
            min_support = min(support.values()) if support else 0
            model_rows.append(
                {
                    "stage": stage,
                    "scheme": scheme,
                    "n_fit": int(len(fit)),
                    "n_valid": int(len(valid)),
                    "accuracy": acc,
                    "macro_f1": f1m,
                    "weighted_f1": f1w,
                    "binary_ap_same_features": ap_bin,
                    "min_class_support_valid": min_support,
                    "support": json.dumps(support),
                    "per_class_f1": json.dumps({c: float(v) for c, v in zip(present, per)}),
                }
            )
            reports[f"{stage}_{scheme}"] = {
                "classification_report": classification_report(y_valid, pred, labels=present, zero_division=0),
                "confusion": cm.tolist(),
                "labels": present,
                "support": support,
            }

    models = pd.DataFrame(model_rows)
    models.to_csv(REP / "multiclass_lr_static.csv", index=False)
    (REP / "multiclass_reports.json").write_text(json.dumps(reports, indent=2), encoding="utf-8")

    lines = [
        "# Khả thi đa lớp OULAD (không đổi bài toán khóa)",
        "",
        "Thử nghiệm **CPU**, Logistic Regression static-only, inner fold 0, FIT-only encode. **Không** train Hybrid, **không** mở outer.",
        "Bài khóa vẫn nhị phân Fail|Withdrawn. Đây chỉ là feasibility.",
        "",
        "## 1. Cỡ mẫu theo cutoff (sau lọc unregistration)",
        "",
        counts.to_string(index=False),
        "",
        "Withdrawn **bốc hơi** khi cutoff tăng: còn hạn nộp thì nhiều người đã rút trước đó bị loại khỏi risk-set.",
        "",
        "## 2. Baseline đa lớp (static LR, fold 0 VALID)",
        "",
        models.drop(columns=["support", "per_class_f1"]).to_string(index=False),
        "",
        "## 3. Kết luận khả thi",
        "",
    ]
    # decision rules
    wd75 = int(counts.loc[counts.stage == "75pct", "Withdrawn"].iloc[0])
    wd35 = int(counts.loc[counts.stage == "35pct", "Withdrawn"].iloc[0])
    f1s = models[models.scheme == "4class"]["macro_f1"]
    lines += [
        f"- 4 lớp (Distinction/Pass/Fail/Withdrawn): Distinction vốn ít; Withdrawn VALID ở 75% còn **{wd75}** trên toàn risk-set (toàn bộ cutoff, chưa split) — sau group-split VALID còn nhỏ hơn ~1/3.",
        f"- Withdrawn 35% còn **{wd35}** (toàn risk-set) — 3 lớp PassBand/Fail/Withdrawn **khả thi hơn** 4 lớp.",
        f"- Macro-F1 4-class trên static LR (fold 0): {', '.join(f'{r.stage}={r.macro_f1:.3f}' for r in models[models.scheme=='4class'].itertuples())}.",
        f"- Macro-F1 3-class: {', '.join(f'{r.stage}={r.macro_f1:.3f}' for r in models[models.scheme=='3class_passband'].itertuples())}.",
        "- Static LR đa lớp **không** thay Hybrid; chỉ cho biết lớp có tách được trên ngữ cảnh tĩnh hay không.",
        "",
        "**Khả thi?**",
        "",
        "- **3 lớp PassBand / Fail / Withdrawn trên 35–50%:** có thể thử (Withdrawn còn đủ). 75% Withdrawn quá mỏng → F1 lớp đó không ổn định.",
        "- **4 lớp (tách Distinction):** kém khả thi — Distinction ít, dễ bị nuốt vào Pass.",
        "- **Không nên thay bài khóa** nếu mục tiêu vẫn là cảnh báo nguy cơ: nhị phân Fail|Withdrawn khớp Rec V (một ngưỡng t). Đa lớp đổi metric (macro-F1), đổi head, đổi Rec V.",
        "- Nếu chỉ **bổ sung phân tích** (không khóa mô hình): 3-class ở 35/50% là mức đáng làm; 4-class và 75% Withdrawn thì không.",
        "",
    ]
    text = "\n".join(lines)
    try:
        md = text
    except Exception:
        md = text
    # to_markdown needs tabulate; fallback
    try:
        (REP / "MULTICLASS_FEASIBILITY.md").write_text(md, encoding="utf-8")
    except Exception:
        (REP / "MULTICLASS_FEASIBILITY.md").write_text(md, encoding="utf-8")
    print("wrote", REP / "MULTICLASS_FEASIBILITY.md")
    print(counts.to_string(index=False))
    print(models[["stage", "scheme", "n_valid", "accuracy", "macro_f1", "min_class_support_valid", "binary_ap_same_features"]].to_string(index=False))


if __name__ == "__main__":
    main()
