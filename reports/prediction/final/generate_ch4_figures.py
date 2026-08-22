"""Generate Chapter 4 figures from locked artifacts only. No invented series."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
FINAL = Path(__file__).resolve().parent
DATA = FINAL / "data_ch4"
FIG = FINAL / "figures"
FIG.mkdir(parents=True, exist_ok=True)

plt.rcParams.update(
    {
        "font.family": "Segoe UI",
        "axes.unicode_minus": False,
        "figure.dpi": 140,
        "savefig.dpi": 160,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
    }
)

UCI_STAGES = ["S0", "S1", "S2"]
OULAD_STAGES = ["20pct", "35pct", "50pct", "75pct", "100pct"]
OULAD_LABELS = ["20%", "35%", "50%", "75%", "100%"]
SERVING = ["Hybrid", "LR", "DT", "RF", "SVM", "MLP", "XGB"]
COLORS = {
    "Hybrid": "#1f4e79",
    "LR": "#7f8c8d",
    "DT": "#bdc3c7",
    "RF": "#e67e22",
    "SVM": "#8e44ad",
    "MLP": "#16a085",
    "XGB": "#c0392b",
    "CatBoost": "#2980b9",
}


def _save(fig: plt.Figure, name: str) -> Path:
    path = FIG / name
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print("wrote", path.relative_to(ROOT))
    return path


def load_serving() -> pd.DataFrame:
    uci = pd.read_csv(FINAL / "uci_final.csv")
    oulad = pd.read_csv(FINAL / "oulad_final.csv")
    return pd.concat([uci, oulad], ignore_index=True)


def fig01_uci_ap_serving(frame: pd.DataFrame) -> Path:
    sub = frame[(frame.dataset == "uci") & (frame.model.isin(SERVING)) & (frame.stage.isin(UCI_STAGES))]
    fig, ax = plt.subplots(figsize=(8.2, 4.4))
    x = np.arange(len(UCI_STAGES))
    width = 0.11
    for i, model in enumerate(SERVING):
        vals = [float(sub[(sub.model == model) & (sub.stage == st)].pr_auc.iloc[0]) for st in UCI_STAGES]
        ax.bar(x + (i - 3) * width, vals, width, label=model, color=COLORS[model])
    ax.set_xticks(x)
    ax.set_xticklabels(UCI_STAGES)
    ax.set_ylabel("AP")
    ax.set_ylim(0.35, 1.0)
    ax.set_title("Hybrid CNN–BiLSTM UCI — AP theo mốc thông tin (3×3); cột khác là bộ so sánh")
    ax.legend(ncol=4, fontsize=8)
    return _save(fig, "fig01_uci_ap_serving.png")


def fig02_oulad_ap_serving(frame: pd.DataFrame) -> Path:
    sub = frame[(frame.dataset == "oulad") & (frame.model.isin(SERVING)) & (frame.stage.isin(OULAD_STAGES))]
    fig, ax = plt.subplots(figsize=(9.2, 4.6))
    x = np.arange(len(OULAD_STAGES))
    width = 0.11
    for i, model in enumerate(SERVING):
        vals = [float(sub[(sub.model == model) & (sub.stage == st)].pr_auc.iloc[0]) for st in OULAD_STAGES]
        ax.bar(x + (i - 3) * width, vals, width, label=model, color=COLORS[model])
    ax.set_xticks(x)
    ax.set_xticklabels(OULAD_LABELS)
    ax.set_ylabel("AP")
    ax.set_ylim(0.65, 0.95)
    ax.set_title("Hybrid CNN–BiLSTM OULAD — AP theo cutoff (3×3); cột khác là bộ so sánh")
    ax.legend(ncol=4, fontsize=8)
    return _save(fig, "fig02_oulad_ap_serving.png")


def fig03_information_growth() -> Path:
    growth = pd.read_csv(FINAL / "information_growth.csv")
    fig, ax = plt.subplots(figsize=(8.4, 4.4))
    uci = growth[growth.dataset == "uci"]
    oulad = growth[growth.dataset == "oulad"]
    ax.plot(uci.delta, uci.pr_auc, marker="o", color=COLORS["Hybrid"], label="UCI ΔAP")
    ax.plot(oulad.delta, oulad.pr_auc, marker="s", color=COLORS["RF"], label="OULAD ΔAP")
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_ylabel("Δ AP")
    ax.set_title("Tăng AP khi thêm thông tin (cùng checkpoint Hybrid, không outer)")
    ax.legend()
    plt.setp(ax.get_xticklabels(), rotation=25, ha="right")
    return _save(fig, "fig03_information_growth_ap.png")


def fig04_overfit() -> Path:
    audit = json.loads((ROOT / "artifacts/prediction/final/OVERFIT_AUDIT.json").read_text(encoding="utf-8"))
    labels, valid, train, gap, cls = [], [], [], [], []
    for name, key in (("UCI", "uci"), ("OULAD", "oulad")):
        stages = ("S0", "S1", "S2") if key == "uci" else OULAD_STAGES
        lab = UCI_STAGES if key == "uci" else OULAD_LABELS
        for st, pretty in zip(stages, lab):
            row = audit[key]["stages"][st]
            labels.append(f"{name} {pretty}")
            valid.append(row["pr_auc_mean"])
            train.append(row["train_pr_auc_mean"])
            gap.append(row["generalization_gap_mean"])
            cls.append(row["overfit_class"])
    fig, ax = plt.subplots(figsize=(9.6, 4.6))
    x = np.arange(len(labels))
    ax.plot(x, train, marker="o", label="AP FIT", color="#c0392b")
    ax.plot(x, valid, marker="s", label="AP VALID (3×3)", color=COLORS["Hybrid"])
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_ylabel("AP")
    ax.set_title("AP_FIT vs AP_VALID — 9 run/mốc. S0 khe 0.125 (HIGH); OULAD LOW")
    for i, (g, c) in enumerate(zip(gap, cls)):
        ax.annotate(f"{g:.3f}\n{c}", (i, valid[i]), textcoords="offset points", xytext=(0, 8), ha="center", fontsize=7)
    ax.legend()
    ax.set_ylim(0.4, 1.02)
    return _save(fig, "fig04_overfit_fit_vs_valid.png")


def fig05_hybrid_uci_metrics(frame: pd.DataFrame) -> Path:
    sub = frame[(frame.dataset == "uci") & (frame.model == "Hybrid") & (frame.stage.isin(UCI_STAGES))]
    metrics = ["accuracy", "pr_auc", "precision", "f1", "recall"]
    names = ["Acc", "AP", "Prec", "F1", "Rec"]
    fig, ax = plt.subplots(figsize=(7.6, 4.4))
    x = np.arange(len(UCI_STAGES))
    width = 0.15
    palette = ["#1f4e79", "#2e86ab", "#a23b72", "#f18f01", "#c73e1d"]
    for i, (col, lab, color) in enumerate(zip(metrics, names, palette)):
        vals = [float(sub[sub.stage == st][col].iloc[0]) for st in UCI_STAGES]
        ax.bar(x + (i - 2) * width, vals, width, label=lab, color=color)
    ax.set_xticks(x)
    ax.set_xticklabels(UCI_STAGES)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Giá trị")
    ax.set_title("Hybrid CNN–BiLSTM UCI — Acc/AP/Prec/F1/Rec tại t STOP (3×3)")
    ax.legend(ncol=5, fontsize=8)
    return _save(fig, "fig05_uci_hybrid_five_metrics.png")


def fig06_hybrid_oulad_metrics(frame: pd.DataFrame) -> Path:
    sub = frame[(frame.dataset == "oulad") & (frame.model == "Hybrid") & (frame.stage.isin(OULAD_STAGES))]
    fig, ax = plt.subplots(figsize=(8.8, 4.5))
    x = np.arange(len(OULAD_STAGES))
    ax.plot(x, [float(sub[sub.stage == st].pr_auc.iloc[0]) for st in OULAD_STAGES], marker="o", label="AP")
    ax.plot(x, [float(sub[sub.stage == st].accuracy.iloc[0]) for st in OULAD_STAGES], marker="s", label="Acc")
    ax.plot(x, [float(sub[sub.stage == st].f1.iloc[0]) for st in OULAD_STAGES], marker="^", label="F1")
    ax.plot(x, [float(sub[sub.stage == st].precision.iloc[0]) for st in OULAD_STAGES], marker="d", label="Prec")
    ax.plot(x, [float(sub[sub.stage == st].recall.iloc[0]) for st in OULAD_STAGES], marker="v", label="Rec")
    ax.set_xticks(x)
    ax.set_xticklabels(OULAD_LABELS)
    ax.set_ylabel("Giá trị")
    ax.set_ylim(0.55, 0.95)
    ax.set_title("Hybrid CNN–BiLSTM OULAD — cùng checkpoint, 5 mốc (3×3)")
    ax.legend(ncol=5, fontsize=8)
    return _save(fig, "fig06_oulad_hybrid_curves.png")


def fig07_ece() -> Path:
    frame = load_serving()
    hy = frame[(frame.model == "Hybrid") & frame.ece.notna()]
    fig, ax = plt.subplots(figsize=(7.8, 4.2))
    labels = [f"UCI {s}" for s in UCI_STAGES] + [f"OULAD {s}" for s in OULAD_LABELS]
    vals = [float(hy[(hy.dataset == "uci") & (hy.stage == s)].ece.iloc[0]) for s in UCI_STAGES]
    vals += [float(hy[(hy.dataset == "oulad") & (hy.stage == s)].ece.iloc[0]) for s in OULAD_STAGES]
    colors = ["#c0392b" if v >= 0.15 else "#e67e22" if v >= 0.08 else COLORS["Hybrid"] for v in vals]
    ax.bar(range(len(labels)), vals, color=colors)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_ylabel("ECE")
    ax.set_title("ECE Hybrid (chỉ bản khóa ghi ECE cho Hybrid). S0 = 0.254; OULAD 100% = 0.020")
    return _save(fig, "fig07_hybrid_ece.png")


def fig08_parity_uci() -> Path:
    frame = load_serving()
    g = frame[(frame.dataset == "uci") & (frame.model.isin(SERVING))]
    fig, ax = plt.subplots(figsize=(8.4, 4.5))
    x = np.arange(3)
    width = 0.11
    for i, model in enumerate(SERVING):
        vals = [float(g[(g.model == model) & (g.stage == st)].pr_auc.iloc[0]) for st in UCI_STAGES]
        ax.bar(x + (i - 3) * width, vals, width, label=model, color=COLORS.get(model, "#555"))
    ax.set_xticks(x)
    ax.set_xticklabels(UCI_STAGES)
    ax.set_ylabel("AP")
    ax.set_title("Hybrid CNN–BiLSTM UCI — AP khóa 3×3 (S1/S2 là claim chính); bộ so sánh cùng protocol")
    ax.legend(ncol=4, fontsize=7)
    ax.set_ylim(0.35, 0.95)
    return _save(fig, "fig08_parity_uci_ap.png")


def fig09_parity_oulad() -> Path:
    frame = load_serving()
    g = frame[(frame.dataset == "oulad") & (frame.model.isin(SERVING))]
    fig, ax = plt.subplots(figsize=(9.2, 4.6))
    x = np.arange(5)
    width = 0.11
    for i, model in enumerate(SERVING):
        vals = [float(g[(g.model == model) & (g.stage == st)].pr_auc.iloc[0]) for st in OULAD_STAGES]
        ax.bar(x + (i - 3) * width, vals, width, label=model, color=COLORS.get(model, "#555"))
    ax.set_xticks(x)
    ax.set_xticklabels(OULAD_LABELS)
    ax.set_ylabel("AP")
    ax.set_ylim(0.65, 0.95)
    ax.set_title("Hybrid CNN–BiLSTM OULAD — AP khóa tăng theo cutoff; một checkpoint, bộ so sánh cùng protocol")
    ax.legend(ncol=4, fontsize=7)
    return _save(fig, "fig09_parity_oulad_ap.png")


def fig10_fold_stop_ap() -> Path:
    payload = json.loads((DATA / "hybrid_oof_fold_reports.json").read_text(encoding="utf-8"))
    stages = ["20pct", "35pct", "50pct", "75pct"]
    fig, ax = plt.subplots(figsize=(7.8, 4.4))
    for report in payload["fold_reports"]:
        ys = [report[f"{st}_valid_pr_auc"] for st in stages]
        ax.plot(OULAD_LABELS[:4], ys, marker="o", label=f"inner fold {report['fold']} seed 42")
    ax.set_ylabel("AP VALID (STOP macro-AP early-stop)")
    ax.set_title("OULAD materialize OOF — AP VALID 4 mốc early, 3 inner fold, seed 42")
    ax.legend()
    ax.set_ylim(0.72, 0.92)
    return _save(fig, "fig10_oulad_fold_stop_ap.png")


def fig11_thresholds() -> Path:
    payload = json.loads((DATA / "hybrid_oof_fold_reports.json").read_text(encoding="utf-8"))
    stages = ["20pct", "35pct", "50pct", "75pct"]
    fig, ax = plt.subplots(figsize=(7.8, 4.4))
    x = np.arange(4)
    width = 0.25
    for i, report in enumerate(payload["fold_reports"]):
        ts = [report[f"{st}_threshold"] for st in stages]
        ax.bar(x + (i - 1) * width, ts, width, label=f"fold {report['fold']}")
    ax.set_xticks(x)
    ax.set_xticklabels(OULAD_LABELS[:4])
    ax.set_ylabel("Ngưỡng t (chọn trên STOP)")
    ax.set_title("t STOP theo fold — fold 2 / 50% t=0.18 vs fold 1 / 75% t=0.52")
    ax.legend()
    return _save(fig, "fig11_stop_threshold_by_fold.png")


def fig12_score_hist() -> Path:
    oof = pd.read_parquet(ROOT / "artifacts/recommend_hybrid/v3/data/c0_oof_predictions.parquet")
    fig, axes = plt.subplots(2, 2, figsize=(8.8, 6.4), sharex=True)
    for ax, st, lab in zip(axes.ravel(), ["20pct", "35pct", "50pct", "75pct"], OULAD_LABELS[:4]):
        p = oof.loc[oof.stage_or_endpoint == st, "risk_probability"]
        ax.hist(p, bins=40, color=COLORS["Hybrid"], alpha=0.85)
        t = float(oof.loc[oof.stage_or_endpoint == st, "prediction_threshold"].median())
        ax.axvline(t, color="#c0392b", linestyle="--", label=f"median t={t:.2f}")
        ax.set_title(f"{lab}  n={len(p):,}  mean p={p.mean():.3f}")
        ax.legend(fontsize=8)
    fig.suptitle("Phân bố p Hybrid trên OOF VALID (66 685 dòng, 3 fold seed 42) — không có nhãn trong file này")
    fig.supxlabel("p = σ(z)")
    return _save(fig, "fig12_oof_score_hist.png")


def fig13_p_vs_uncertainty() -> Path:
    oof = pd.read_parquet(ROOT / "artifacts/recommend_hybrid/v3/data/c0_oof_predictions.parquet")
    sample = oof.sample(n=min(8000, len(oof)), random_state=42)
    fig, ax = plt.subplots(figsize=(7.2, 5.2))
    sc = ax.scatter(sample.risk_probability, sample.uncertainty, c=sample.inner_fold, s=6, alpha=0.35, cmap="viridis")
    ax.set_xlabel("p")
    ax.set_ylabel("H₂(p)")
    ax.set_title("p vs entropy nhị phân — mẫu 8000 OOF. H₂ max tại p=0.5")
    fig.colorbar(sc, ax=ax, label="inner fold")
    return _save(fig, "fig13_p_vs_entropy.png")


def fig14_rec_metrics() -> Path:
    rec = json.loads((DATA / "PANEL_C_FINAL_RESULTS.json").read_text(encoding="utf-8"))
    labels = ["Recommendation V", "B1 rule", "B0 action+stage"]
    keys = ["five_ebm_c0", "baseline_b1", "baseline_b0"]
    ndcg = [rec[k]["ndcg_at_3"] for k in keys]
    p1 = [rec[k]["precision_at_1"] for k in keys]
    r3 = [rec[k]["recall_at_3"] for k in keys]
    x = np.arange(3)
    fig, ax = plt.subplots(figsize=(7.6, 4.4))
    ax.bar(x - 0.25, ndcg, 0.25, label="NDCG@3", color=COLORS["Hybrid"])
    ax.bar(x, p1, 0.25, label="P@1", color=COLORS["RF"])
    ax.bar(x + 0.25, r3, 0.25, label="R@3", color=COLORS["CatBoost"])
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0.7, 1.02)
    ax.set_title("Panel C held-out 632 case — NDCG@3 V = 0.88785 vs B1 0.86649")
    ax.legend()
    return _save(fig, "fig14_rec_panel_c_metrics.png")


def fig15_routes() -> Path:
    rec = json.loads((DATA / "PANEL_C_FINAL_RESULTS.json").read_text(encoding="utf-8"))
    counts = rec["pipeline_system"]["route_counts"]
    fig, ax = plt.subplots(figsize=(6.8, 4.6))
    labels = list(counts.keys())
    vals = [counts[k] for k in labels]
    ax.bar(labels, vals, color=["#1f4e79", "#7f8c8d", "#e67e22", "#27ae60"])
    for i, v in enumerate(vals):
        ax.text(i, v + 4, f"{v}\n({v/632:.1%})", ha="center", fontsize=8)
    ax.set_ylabel("Số case")
    ax.set_title("Panel C 632 case: INSUFFICIENT_EVIDENCE 363 (57.4%), RECOMMEND 94 (14.9%)")
    plt.setp(ax.get_xticklabels(), rotation=15, ha="right")
    return _save(fig, "fig15_rec_routes.png")


def fig16_top1() -> Path:
    rec = json.loads((DATA / "PANEL_C_FINAL_RESULTS.json").read_text(encoding="utf-8"))
    dist = rec["pipeline_system"]["top1_action_distribution"]
    fig, ax = plt.subplots(figsize=(8.2, 4.4))
    keys = list(dist.keys())
    vals = [dist[k] for k in keys]
    ax.barh(keys, vals, color=COLORS["Hybrid"])
    ax.set_xlabel("Số case (Top-1 trên 632)")
    ax.set_title("Top-1 đủ 5 hành động; RECOVER_ENGAGEMENT 111 / 632")
    return _save(fig, "fig16_rec_top1_actions.png")


def fig17_bootstrap() -> Path:
    boot = json.loads((DATA / "PANEL_C_BOOTSTRAP.json").read_text(encoding="utf-8"))
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    rows = [("V − B0", boot["v3_minus_b0"]), ("V − B1", boot["v3_minus_b1"])]
    y = np.arange(len(rows))
    means = [r[1]["mean_difference"] for r in rows]
    lo = [r[1]["ci_low_95"] for r in rows]
    hi = [r[1]["ci_high_95"] for r in rows]
    ax.errorbar(means, y, xerr=[np.array(means) - np.array(lo), np.array(hi) - np.array(means)], fmt="o", color=COLORS["Hybrid"], capsize=5)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_yticks(y)
    ax.set_yticklabels([r[0] for r in rows])
    ax.set_xlabel("Δ NDCG@3 (bootstrap 2000, seed 2026)")
    ax.set_title("V−B1: +0.0213, 95% CI [0.0144, 0.0282], P(Δ>0)=1.00")
    return _save(fig, "fig17_rec_bootstrap_ndcg.png")


def fig18_ap_errorbar() -> Path:
    audit = json.loads((ROOT / "artifacts/prediction/final/OVERFIT_AUDIT.json").read_text(encoding="utf-8"))
    fig, ax = plt.subplots(figsize=(8.6, 4.4))
    labels, means, stds = [], [], []
    for pretty, st in zip(UCI_STAGES, UCI_STAGES):
        row = audit["uci"]["stages"][st]
        labels.append(f"UCI {pretty}")
        means.append(row["pr_auc_mean"])
        stds.append(row["pr_auc_std"])
    for pretty, st in zip(OULAD_LABELS, OULAD_STAGES):
        row = audit["oulad"]["stages"][st]
        labels.append(f"OULAD {pretty}")
        means.append(row["pr_auc_mean"])
        stds.append(row["pr_auc_std"])
    x = np.arange(len(labels))
    ax.errorbar(x, means, yerr=stds, fmt="o-", color=COLORS["Hybrid"], capsize=4)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_ylabel("AP VALID mean ± std (n=9)")
    ax.set_title("Phương sai 3 fold × 3 seed. UCI S0 std=0.043; OULAD std ≤ 0.008")
    return _save(fig, "fig18_ap_mean_std_9run.png")


def main() -> list[Path]:
    serving = load_serving()
    paths = [
        fig01_uci_ap_serving(serving),
        fig02_oulad_ap_serving(serving),
        fig03_information_growth(),
        fig04_overfit(),
        fig05_hybrid_uci_metrics(serving),
        fig06_hybrid_oulad_metrics(serving),
        fig07_ece(),
        fig08_parity_uci(),
        fig09_parity_oulad(),
        fig10_fold_stop_ap(),
        fig11_thresholds(),
        fig12_score_hist(),
        fig13_p_vs_uncertainty(),
        fig14_rec_metrics(),
        fig15_routes(),
        fig16_top1(),
        fig17_bootstrap(),
        fig18_ap_errorbar(),
    ]
    manifest = {"n_figures": len(paths), "files": [p.name for p in paths]}
    (FIG / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print("n_figures", len(paths))
    return paths


if __name__ == "__main__":
    main()
