"""P0.2: Hybrid vs LR/RF on 9 locked runs + dual-AP investigation."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

from .paths import PHASE4, REP, ROOT, ensure

HYBRID_CSV = PHASE4 / "ROBUST_CONFIRMATION.csv"
BASE_CSV = PHASE4 / "BASELINE_INNER_RESULTS.csv"
UCI_FINAL = ROOT / "reports" / "prediction" / "final" / "uci_final.csv"
OULAD_FINAL = ROOT / "reports" / "prediction" / "final" / "oulad_final.csv"
FAIR_UCI = ROOT / "reports" / "prediction" / "final" / "data_ch4" / "baseline_fair_stage_metrics_uci.csv"
FAIR_OULAD = ROOT / "reports" / "prediction" / "final" / "data_ch4" / "baseline_fair_stage_metrics_oulad.csv"
FIG_SCRIPT = ROOT / "reports" / "prediction" / "final" / "generate_ch4_figures.py"


def paired(hybrid: pd.Series, other: pd.Series) -> dict:
    delta = hybrid.to_numpy() - other.to_numpy()
    n = len(delta)
    if n < 2 or np.allclose(delta, 0):
        p = float("nan")
    else:
        try:
            p = float(wilcoxon(hybrid, other, alternative="two-sided").pvalue)
        except ValueError:
            p = float("nan")
    rng = np.random.default_rng(2026)
    boots = []
    for _ in range(2000):
        idx = rng.integers(0, n, n)
        boots.append(float(delta[idx].mean()))
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return {
        "n": n,
        "mean_hybrid": float(hybrid.mean()),
        "mean_other": float(other.mean()),
        "mean_delta": float(delta.mean()),
        "ci95_low": float(lo),
        "ci95_high": float(hi),
        "wilcoxon_p": p,
        "hybrid_gt_count": int((delta > 0).sum()),
    }


def main() -> None:
    ensure()
    hy = pd.read_csv(HYBRID_CSV)
    hy = hy[hy.strategy == "L1_control"].copy()
    base = pd.read_csv(BASE_CSV)
    lines = [
        "# STAT_SIGNIFICANCE",
        "",
        "Nguồn Hybrid 9-run: `test_lab/artifacts/hybrid_vnext/phase4/ROBUST_CONFIRMATION.csv` (L1_control).",
        "Nguồn baseline: `BASELINE_INNER_RESULTS.csv`.",
        "Kiểm định: Wilcoxon signed-rank trên 9 cặp (fold, seed), bootstrap 2000 trên hiệu AP. **Không** mở outer.",
        "",
        "| domain | stage | vs | Hybrid AP | other AP | Δ | 95% CI | Wilcoxon p | Hybrid>other |",
        "|---|---|---|---:|---:|---:|---|---:|---:|",
    ]
    rows = []
    for dataset in ("uci", "oulad"):
        stages = ["S0", "S1", "S2"] if dataset == "uci" else ["20pct", "35pct", "50pct", "75pct", "100pct"]
        for stage in stages:
            h = hy[(hy.dataset == dataset) & (hy.stage == stage)].sort_values(["inner_fold", "seed"])
            if len(h) != 9:
                lines.append(f"| {dataset} | {stage} | — | TODO n={len(h)} not 9 | | | | | |")
                continue
            for family in ("LR", "RF"):
                b = base[(base.dataset == dataset) & (base.stage == stage) & (base.family == family)].sort_values(
                    ["inner_fold", "seed"]
                )
                merged = pd.merge(
                    h[["inner_fold", "seed", "pr_auc"]],
                    b[["inner_fold", "seed", "pr_auc"]],
                    on=["inner_fold", "seed"],
                    suffixes=("_h", "_b"),
                )
                stats = paired(merged.pr_auc_h, merged.pr_auc_b)
                stats.update({"dataset": dataset, "stage": stage, "family": family})
                rows.append(stats)
                lines.append(
                    f"| {dataset} | {stage} | {family} | {stats['mean_hybrid']:.4f} | {stats['mean_other']:.4f} | "
                    f"{stats['mean_delta']:+.4f} | [{stats['ci95_low']:+.4f}, {stats['ci95_high']:+.4f}] | "
                    f"{stats['wilcoxon_p']:.4g} | {stats['hybrid_gt_count']}/{stats['n']} |"
                )
    (REP / "stat_pairs.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")

    serving_uci = pd.read_csv(UCI_FINAL)
    serving_h = serving_uci[serving_uci.model == "Hybrid"].set_index("stage")["pr_auc"]
    lines += [
        "",
        "## Hai số AP UCI S1: 0.821 vs 0.811",
        "",
        f"- Serving 3×3 (`uci_final.csv` cột `pr_auc`): S0={serving_h.get('S0')}, S1={serving_h.get('S1')}, S2={serving_h.get('S2')}.",
        f"- Cùng file OVERFIT_AUDIT / Chương 4 khóa: S1 **{float(serving_h['S1']):.4f}**.",
        f"- Robust L1_control mean S1 from ROBUST_CONFIRMATION: **{hy[(hy.dataset=='uci')&(hy.stage=='S1')].pr_auc.mean():.4f}** (should match serving if same 9 jobs).",
    ]
    if FAIR_UCI.exists():
        fair = pd.read_csv(FAIR_UCI)
        lines.append(
            f"- Tensor-parity CSV `{FAIR_UCI.name}` models={sorted(fair.model.unique())}. **Không có hàng Hybrid** — số 0.811 không thể lấy từ file này."
        )
    text = FIG_SCRIPT.read_text(encoding="utf-8") if FIG_SCRIPT.exists() else ""
    lines += [
        "- Số **0.8110 / 0.9132** được hard-code trong `generate_ch4_figures.py` (`hybrid = {S0: 0.4559, S1: 0.8110, S2: 0.9132}`), comment 'from CHUONG_3 3.3.6'. Đó là panel tensor-parity cũ, **không** phải 9-run serving.",
        "- **Số khóa để báo cáo serving:** UCI S1 AP **0.8214** (9-run L1). Số 0.811 chỉ dùng khi nói panel cùng-tensor (và phải ghi nguồn hard-code / thiếu hàng Hybrid trong CSV).",
        "",
        "Nếu CI 95% chứa 0 hoặc p>0.05: Hybrid **không** hơn baseline có ý nghĩa thống kê trên mốc đó. Viết đúng như vậy ở Chương 4/5.",
        "",
    ]
    (REP / "STAT_SIGNIFICANCE.md").write_text("\n".join(lines), encoding="utf-8")
    print("wrote", REP / "STAT_SIGNIFICANCE.md")


if __name__ == "__main__":
    main()
