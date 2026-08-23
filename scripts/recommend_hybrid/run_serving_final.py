"""Train and evaluate the locked persistence recommendation model.

Labels use 14-day OULAD logs only. final_result is evaluation-only.
Hybrid probabilities are frozen OOF values in learner_stage_features.parquet.
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.recommend_hybrid.serving.contracts import FEATURE_COLUMNS, K_FRAC_PRIMARY, PROTOCOL_VERSION
from src.recommend_hybrid.serving.labels import attach_persistence_labels
from src.recommend_hybrid.serving.metrics import (
    feasibility_audit,
    model_scores,
    targeting_table,
    tier4_block,
)
from src.recommend_hybrid.serving.model import (
    CLASSES,
    PersistenceClassifier,
    feature_matrix,
    fit_select,
    rule_predict,
)
from src.recommend_hybrid.serving.policy import attach_worklist

FEATURES = ROOT / "artifacts" / "recommend_hybrid" / "v3" / "data" / "learner_stage_features.parquet"
RAW = ROOT / "data" / "raw"
OUT = ROOT / "artifacts" / "recommend_hybrid" / "serving"
FIG = ROOT / "reports" / "prediction" / "final" / "figures"
REPORT_DATA = ROOT / "reports" / "prediction" / "final" / "data_ch4"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def log(message: str) -> None:
    print(f"[{utc_now()}] {message}", flush=True)


def load_base() -> pd.DataFrame:
    frame = pd.read_parquet(FEATURES)
    info = pd.read_csv(RAW / "studentInfo.csv")
    info["id_student"] = info["id_student"].astype(str)
    frame["id_student"] = frame["id_student"].astype(str)
    info = info[["id_student", "code_module", "code_presentation", "final_result"]].drop_duplicates()
    frame = frame.merge(info, on=["id_student", "code_module", "code_presentation"], how="left")
    frame["y"] = frame["final_result"].isin(["Fail", "Withdrawn"]).astype(int)
    return frame


def split_students(frame: pd.DataFrame, seed: int = 2026) -> pd.DataFrame:
    students = frame["id_student"].astype(str).unique()
    y_student = (
        frame.groupby("id_student")["y"].max().reindex(students).fillna(0).astype(int).to_numpy()
    )
    gss = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=seed)
    dummy = np.zeros(len(students))
    train_val_idx, test_idx = next(gss.split(dummy, y_student, groups=students))
    rest = students[train_val_idx]
    y_rest = y_student[train_val_idx]
    gss2 = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=seed + 1)
    tr_idx, va_idx = next(gss2.split(np.zeros(len(rest)), y_rest, groups=rest))
    mapping = {sid: "test" for sid in students[test_idx]}
    mapping.update({sid: "train" for sid in rest[tr_idx]})
    mapping.update({sid: "val" for sid in rest[va_idx]})
    out = frame.copy()
    out["split"] = out["id_student"].astype(str).map(mapping)
    return out


def draw_figures(target: pd.DataFrame, scores: dict, t4: dict, out_dir: Path) -> list[str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir.mkdir(parents=True, exist_ok=True)
    saved = []
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    stages = target["stage"].tolist()
    x = np.arange(len(stages))
    w = 0.35
    ax.bar(x - w / 2, target["precision@10"], w, label="Precision@10%", color="#1f4e79")
    ax.bar(x + w / 2, target["recall@10"], w, label="Recall@10%", color="#7aa2c4")
    ax.set_xticks(x)
    ax.set_xticklabels([str(s).replace("EARLY_", "").replace("MIDDLE_", "").replace("LATE_", "") for s in stages])
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Chỉ số")
    ax.set_title("Tầng 1. Hàng đợi top 10% theo p (Hybrid khóa)")
    ax.legend(frameon=False)
    fig.tight_layout()
    p1 = out_dir / "fig_rec_targeting.png"
    fig.savefig(p1, dpi=160)
    plt.close(fig)
    saved.append(str(p1.name))

    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    names = ["Mô hình rec", "Luật đuôi"]
    f1s = [scores["model"]["macro_f1"], scores["rule"]["macro_f1"]]
    aps = [scores["model"].get("macro_ap", 0), scores["rule"].get("macro_ap", 0)]
    x = np.arange(2)
    ax.bar(x - 0.18, f1s, 0.36, label="Macro-F1", color="#1f4e79")
    ax.bar(x + 0.18, aps, 0.36, label="Macro-AP", color="#c45911")
    ax.set_xticks(x)
    ax.set_xticklabels(names)
    ax.set_ylim(0, 1.05)
    ax.set_title("Kết quả mô hình rec (nhãn tồn tại 14 ngày, tập test)")
    ax.legend(frameon=False)
    fig.tight_layout()
    p2 = out_dir / "fig_rec_model.png"
    fig.savefig(p2, dpi=160)
    plt.close(fig)
    saved.append(str(p2.name))

    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    routes = scores.get("routes", {})
    labels = list(routes.keys()) or ["ACTION", "QUEUE", "COUNSEL", "OUT_OF_BUDGET"]
    vals = [routes.get(k, 0) for k in labels]
    ax.bar(labels, vals, color=["#1f4e79", "#7aa2c4", "#c45911", "#9e9e9e"])
    ax.set_ylabel("Tỷ lệ")
    ax.set_title("Trạng thái phát hành trên tập test")
    ax.set_ylim(0, 1)
    fig.tight_layout()
    p3 = out_dir / "fig_rec_routes.png"
    fig.savefig(p3, dpi=160)
    plt.close(fig)
    saved.append(str(p3.name))

    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    matched = t4.get("matched", {})
    mismatched = t4.get("mismatched", {})
    ax.bar(
        ["Khớp cơ chế", "Lệch cơ chế"],
        [matched.get("beta1_mean", 0), mismatched.get("beta1_mean", 0)],
        color=["#1f4e79", "#c45911"],
    )
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_ylabel("β1 (bootstrap mean)")
    ax.set_title("Tầng 4. Đặc hiệu cơ chế (kiểm soát p)")
    fig.tight_layout()
    p4 = out_dir / "fig_rec_specificity.png"
    fig.savefig(p4, dpi=160)
    plt.close(fig)
    saved.append(str(p4.name))
    return saved


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    labeled_path = OUT / "labeled_queries.parquet"
    log("load Hybrid OOF features")
    base = load_base()
    log(f"features n={len(base)} students={base['id_student'].nunique()}")
    if labeled_path.is_file():
        log(f"reuse labels {labeled_path}")
        labeled = pd.read_parquet(labeled_path)
    else:
        log("build 14-day persistence labels (studentVle scan)")
        labeled = attach_persistence_labels(base, RAW)
        labeled.to_parquet(labeled_path, index=False)
        log(f"wrote {labeled_path}")
    labeled = split_students(labeled)
    labeled = attach_worklist(labeled, k_frac=K_FRAC_PRIMARY)
    for column in FEATURE_COLUMNS:
        if column not in labeled.columns:
            labeled[column] = np.nan
    train = labeled.loc[labeled["split"].eq("train")].copy()
    val = labeled.loc[labeled["split"].eq("val")].copy()
    test = labeled.loc[labeled["split"].eq("test")].copy()
    log(f"split train={len(train)} val={len(val)} test={len(test)}")
    log("fit persistence models")
    bundle = fit_select(
        feature_matrix(train),
        train["persist_label"].to_numpy(),
        feature_matrix(val),
        val["persist_label"].to_numpy(),
    )
    clf = PersistenceClassifier(bundle)
    model_path = OUT / "persistence_classifier.joblib"
    clf.save(model_path)
    log(f"selected {bundle['selected']} val_macro_ap={bundle['val_macro_ap']:.4f}")

    test_actions, test_scores = clf.constrained_predict(test)
    test = test.copy()
    test["pred_action"] = test_actions
    test["pred_score"] = test_scores
    test["rule_action"] = rule_predict(test)
    proba = clf.predict_proba(test)
    model_m = model_scores(test["persist_label"].to_numpy(), test["pred_action"].to_numpy(), proba)
    rule_m = model_scores(test["persist_label"].to_numpy(), test["rule_action"].to_numpy())
    work = test.loc[test["in_worklist"]].copy()
    work_model = model_scores(work["persist_label"].to_numpy(), work["pred_action"].to_numpy(), clf.predict_proba(work)) if len(work) else {}
    work_rule = model_scores(work["persist_label"].to_numpy(), work["rule_action"].to_numpy()) if len(work) else {}
    from src.recommend_hybrid.serving.policy import route_for_row

    routes = test.apply(lambda r: route_for_row(r, str(r["pred_action"])).value, axis=1)
    test["route"] = routes
    route_share = routes.value_counts(normalize=True).to_dict()
    target = targeting_table(labeled)
    audit = feasibility_audit(test)
    t4 = tier4_block(test)
    beats_rule = float(model_m["macro_f1"]) > float(rule_m["macro_f1"])
    payload = {
        "protocol_version": PROTOCOL_VERSION,
        "created_at_utc": utc_now(),
        "selected_model": bundle["selected"],
        "val": {"macro_ap": bundle["val_macro_ap"], "macro_f1": bundle["val_macro_f1"]},
        "test_all": {"model": model_m, "rule": rule_m, "beats_rule": beats_rule},
        "test_worklist": {"model": work_model, "rule": work_rule, "n": int(len(work))},
        "routes": route_share,
        "targeting": target.to_dict(orient="records"),
        "feasibility": audit,
        "tier4": t4,
        "split_counts": labeled["split"].value_counts().to_dict(),
        "label_counts": labeled["persist_label"].value_counts().to_dict(),
        "hybrid_locked": True,
        "causal_claim": False,
        "model_path": str(model_path.relative_to(ROOT)).replace("\\", "/"),
    }
    OUT.joinpath("SERVING_METRICS.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    REPORT_DATA.mkdir(parents=True, exist_ok=True)
    REPORT_DATA.joinpath("REC_SERVING_METRICS.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    target.to_csv(REPORT_DATA / "rec_targeting.csv", index=False)
    labeled.loc[labeled["split"].eq("test"), ["query_id", "split", "in_worklist", "persist_label"]].to_parquet(
        OUT / "test_keys.parquet", index=False
    )
    test.assign(pred_action=test_actions).to_parquet(OUT / "test_predictions.parquet", index=False)
    cohort_cols = [c for c in labeled.columns if c not in {"final_result"}]
    labeled.loc[:, [c for c in cohort_cols if c != "y"]].to_parquet(OUT / "cohort_features.parquet", index=False)
    figs = draw_figures(target, {"model": model_m, "rule": rule_m, "routes": route_share}, t4, FIG)
    log(f"metrics written; figures={figs}")
    log(json.dumps({"test_macro_f1_model": model_m["macro_f1"], "test_macro_f1_rule": rule_m["macro_f1"], "beats_rule": beats_rule}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
