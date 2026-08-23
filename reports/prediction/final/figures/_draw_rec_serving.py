"""Chapter 3 figure: persistence recommendation on Hybrid top-K."""
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

OUT = Path(__file__).resolve().parent
plt.rcParams.update({"font.family": "Segoe UI", "axes.unicode_minus": False, "figure.dpi": 160})


def box(ax, x, y, w, h, text, color, size=9):
    ax.add_patch(
        FancyBboxPatch(
            (x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.08",
            linewidth=1.1, edgecolor="#1f3a5f", facecolor=color,
        )
    )
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=size, color="#1a1a1a")


def arrow(ax, a, b):
    ax.add_patch(FancyArrowPatch(a, b, arrowstyle="-|>", mutation_scale=12, linewidth=1.1, color="#2c3e50"))


def main():
    fig, ax = plt.subplots(figsize=(11.0, 5.6))
    ax.set_xlim(0, 11)
    ax.set_ylim(0, 5.6)
    ax.axis("off")
    ax.set_title("Module khuyến nghị: hàng đợi top-K và nút thắt còn kéo dài", fontsize=13, color="#1f3a5f")
    box(ax, 0.3, 3.5, 2.3, 1.2, "Hybrid khóa\np, t, độ bất định", "#d6eaf8")
    box(ax, 0.3, 1.6, 2.3, 1.2, "Bằng chứng trước τ\nthiếu bài, nghỉ VLE", "#d5f5e3")
    box(ax, 3.2, 2.5, 2.5, 1.4, "Hàng đợi\ntop 10% theo p", "#f9e79f")
    box(ax, 6.2, 3.5, 2.4, 1.2, "Mô hình rec\nP(nút thắt còn 14 ngày)", "#f5b7b1")
    box(ax, 6.2, 1.6, 2.4, 1.2, "Luật khả thi\n+ lộ trình bài còn hạn", "#d2b4de")
    box(ax, 9.0, 2.3, 1.8, 1.6, "ACTION\nQUEUE\nCOUNSEL\nOUT_OF_BUDGET", "#a9cce3", 8.5)
    arrow(ax, (2.6, 4.1), (3.2, 3.4))
    arrow(ax, (2.6, 2.2), (3.2, 3.0))
    arrow(ax, (5.7, 3.2), (6.2, 4.1))
    arrow(ax, (5.7, 3.1), (6.2, 2.2))
    arrow(ax, (8.6, 4.1), (9.0, 3.3))
    arrow(ax, (8.6, 2.2), (9.0, 2.9))
    fig.tight_layout()
    fig.savefig(OUT / "recommendation_architecture.png", dpi=180)
    fig.savefig(OUT / "fig_rec_pipeline.png", dpi=180)


if __name__ == "__main__":
    main()
