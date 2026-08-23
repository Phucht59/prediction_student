"""Scientific architecture figure for Chapter 3. Vietnamese labels, no code names."""
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

OUT = Path(__file__).resolve().parent
plt.rcParams.update(
    {
        "font.family": "Segoe UI",
        "axes.unicode_minus": False,
        "figure.dpi": 160,
        "savefig.dpi": 180,
    }
)


def box(ax, x, y, w, h, text, facecolor, fontsize=9.5, bold=False):
    p = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.02,rounding_size=0.08",
        linewidth=1.1,
        edgecolor="#1f3a5f",
        facecolor=facecolor,
    )
    ax.add_patch(p)
    ax.text(
        x + w / 2,
        y + h / 2,
        text,
        ha="center",
        va="center",
        fontsize=fontsize,
        color="#1a1a1a",
        fontweight="bold" if bold else "normal",
        wrap=True,
    )
    return (x + w / 2, y + h / 2)


def arrow(ax, p1, p2):
    ax.add_patch(
        FancyArrowPatch(
            p1,
            p2,
            arrowstyle="-|>",
            mutation_scale=12,
            linewidth=1.15,
            color="#2c3e50",
            shrinkA=2,
            shrinkB=2,
        )
    )


def draw_architecture():
    fig, ax = plt.subplots(figsize=(11.2, 6.4))
    ax.set_xlim(0, 11.2)
    ax.set_ylim(0, 6.4)
    ax.axis("off")
    ax.set_title("Kiến trúc Hybrid CNN–BiLSTM đề xuất", fontsize=14, color="#1f3a5f", pad=8)

    # inputs
    c_static = box(ax, 0.25, 4.7, 2.15, 1.05, "Đặc trưng tĩnh\n(bối cảnh sinh viên)", "#d6eaf8")
    c_agg = box(ax, 0.25, 3.25, 2.15, 1.05, "Đặc trưng gộp\n(tại mốc quan sát)", "#d5f5e3")
    c_seq = box(ax, 0.25, 1.45, 2.15, 1.25, "Chuỗi thời gian\n(điểm / tuần VLE)\nkèm mặt nạ", "#fdebd0")

    # branches
    b_tab = box(ax, 3.0, 3.85, 2.35, 1.35, "Nhánh bảng\nchiếu residual\nthống kê + ngữ cảnh", "#aed6f1")
    b_cnn = box(ax, 3.0, 2.25, 2.35, 1.2, "Module CNN\n64 kênh, kernel 2\ndilation 1 rồi 2", "#f5b7b1")
    b_lstm = box(ax, 3.0, 0.7, 2.35, 1.2, "Module Bi-LSTM\nẩn 128, một lớp\nhai chiều", "#d2b4de")

    gate = box(ax, 6.05, 2.15, 2.45, 1.7, "Cổng kết hợp\nsoftmax 3 nhánh\n(nhánh không có\nchuỗi bị tắt)", "#f9e79f", fontsize=10)
    head = box(ax, 9.05, 2.35, 1.9, 1.3, "Đầu ra\nxác suất nguy cơ\np = σ(z)", "#a9cce3", fontsize=10)

    rec = box(ax, 6.2, 0.25, 4.7, 0.95, "Module khuyến nghị: xếp hạng hành động hỗ trợ khả thi\n(chỉ dùng xác suất nguy cơ và bằng chứng đã quan sát)", "#fadbd8", fontsize=9)

    arrow(ax, (c_static[0] + 1.08, c_static[1]), (b_tab[0] - 1.18, b_tab[1] + 0.25))
    arrow(ax, (c_agg[0] + 1.08, c_agg[1]), (b_tab[0] - 1.18, b_tab[1] - 0.2))
    arrow(ax, (c_seq[0] + 1.08, c_seq[1] + 0.15), (b_cnn[0] - 1.18, b_cnn[1]))
    arrow(ax, (c_seq[0] + 1.08, c_seq[1] - 0.2), (b_lstm[0] - 1.18, b_lstm[1]))
    arrow(ax, (b_tab[0] + 1.18, b_tab[1]), (gate[0] - 1.23, gate[1] + 0.45))
    arrow(ax, (b_cnn[0] + 1.18, b_cnn[1]), (gate[0] - 1.23, gate[1]))
    arrow(ax, (b_lstm[0] + 1.18, b_lstm[1]), (gate[0] - 1.23, gate[1] - 0.5))
    arrow(ax, (gate[0] + 1.23, gate[1]), (head[0] - 0.95, head[1]))
    arrow(ax, (head[0], head[1] - 0.65), (rec[0] + 1.6, rec[1] + 0.48))

    ax.text(1.32, 6.05, "Đầu vào trước mốc quan sát", ha="center", fontsize=8.5, color="#5d6d7e")
    ax.text(4.17, 5.45, "Ba nhánh song song", ha="center", fontsize=8.5, color="#5d6d7e")
    ax.text(7.27, 4.05, "Kết hợp có điều kiện", ha="center", fontsize=8.5, color="#5d6d7e")

    fig.tight_layout()
    path = OUT / "architecture_hybrid.png"
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("wrote", path)


def draw_flow():
    fig, ax = plt.subplots(figsize=(11.0, 2.6))
    ax.set_xlim(0, 11)
    ax.set_ylim(0, 2.6)
    ax.axis("off")
    ax.set_title("Luồng xử lý từ dữ liệu đến khuyến nghị", fontsize=13, color="#1f3a5f", pad=6)
    labels = [
        (0.2, "Dữ liệu gốc\nUCI / OULAD"),
        (2.35, "Tiền xử lý\nkhông rò rỉ thời gian"),
        (4.5, "Hybrid\nCNN–BiLSTM"),
        (6.65, "Xác suất\nnguy cơ"),
        (8.8, "Module\nkhuyến nghị"),
    ]
    centers = []
    for x, lab in labels:
        centers.append(box(ax, x, 0.55, 1.95, 1.25, lab, "#eaf2f8", fontsize=9.5))
    for i in range(len(centers) - 1):
        arrow(ax, (centers[i][0] + 0.98, centers[i][1]), (centers[i + 1][0] - 0.98, centers[i + 1][1]))
    fig.tight_layout()
    path = OUT / "hybrid_pipeline_flow.png"
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("wrote", path)


if __name__ == "__main__":
    draw_architecture()
    draw_flow()
