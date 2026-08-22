"""P0.5 gate weights from locked GATE_DIAGNOSTICS + serving checkpoints if loadable."""
from __future__ import annotations

import json

import matplotlib.pyplot as plt
import pandas as pd
import torch

from .paths import FIG, PHASE4, REP, SERVING_CKPT, ensure


def main() -> None:
    ensure()
    gate = pd.read_csv(PHASE4 / "GATE_DIAGNOSTICS.csv")
    gate = gate[gate.strategy == "L1_control"].copy()
    summary = (
        gate.groupby(["dataset", "stage"])[["tabular_mass", "cnn_mass", "bilstm_mass"]]
        .mean()
        .reset_index()
    )
    summary.to_csv(REP / "gate_weights_by_cutoff.csv", index=False)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    order_u = ["S0", "S1", "S2"]
    order_o = ["20pct", "35pct", "50pct", "75pct", "100pct"]
    for ax, dataset, order, title in (
        (axes[0], "uci", order_u, "UCI"),
        (axes[1], "oulad", order_o, "OULAD"),
    ):
        sub = summary[summary.dataset == dataset].set_index("stage").reindex(order)
        x = range(len(order))
        ax.plot(x, sub.tabular_mass, marker="o", label="tabular")
        ax.plot(x, sub.cnn_mass, marker="s", label="CNN")
        ax.plot(x, sub.bilstm_mass, marker="^", label="BiLSTM")
        ax.set_xticks(list(x))
        ax.set_xticklabels(order)
        ax.set_ylim(0, 1.05)
        ax.set_title(f"Gate mass — {title} (mean L1 9-run)")
        ax.legend()
        ax.set_ylabel("mean softmax mass")
    fig.tight_layout()
    fig.savefig(FIG / "gate_weights_by_cutoff.png", dpi=160)
    plt.close(fig)

    serving_note = []
    for fold in (0, 1, 2):
        path = SERVING_CKPT / f"c0_inner_fold{fold}_seed42.pt"
        if not path.exists():
            serving_note.append(f"missing {path.name}")
            continue
        payload = torch.load(path, map_location="cpu", weights_only=False)
        serving_note.append(f"{path.name} keys={list(payload)[:8]} model_id={payload.get('model_id')}")

    lines = [
        "# Gate weights by cutoff",
        "",
        "Nguồn chính: `test_lab/artifacts/hybrid_vnext/phase4/GATE_DIAGNOSTICS.csv` (L1_control, 9 run).",
        f"Hình: `{(FIG / 'gate_weights_by_cutoff.png').as_posix()}`.",
        "",
        "| dataset | stage | tabular | CNN | BiLSTM |",
        "|---|---|---:|---:|---:|",
    ]
    for r in summary.itertuples():
        lines.append(f"| {r.dataset} | {r.stage} | {r.tabular_mass:.3f} | {r.cnn_mass:.3f} | {r.bilstm_mass:.3f} |")
    lines += [
        "",
        "UCI S0: tabular_mass = 1 (CNN/BiLSTM tắt, T=0) — đúng thiết kế.",
        "Nếu CNN+BiLSTM mass tăng từ 20%→75% trên OULAD, cổng đang chuyển sang chuỗi khi có tuần VLE.",
        "",
        "Serving checkpoints: " + "; ".join(serving_note),
        "",
    ]
    (REP / "GATE_WEIGHTS.md").write_text("\n".join(lines), encoding="utf-8")
    print("wrote", FIG / "gate_weights_by_cutoff.png")


if __name__ == "__main__":
    main()
