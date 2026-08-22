"""Draft Ch1–5 + SUMMARY + extra architecture figure from research outputs."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pandas as pd

from .paths import CH, FIG, REP, ROOT, ensure


def _read(name: str) -> str:
    path = REP / name
    return path.read_text(encoding="utf-8") if path.exists() else f"TODO — chưa có {name}"


def _ablation_table() -> str:
    csv = ROOT / "artifacts" / "research" / "kltn_science_fix" / "ablation_raw.csv"
    if not csv.exists():
        return "TODO — ablation_raw.csv chưa xong."
    frame = pd.read_csv(csv)
    g = frame.groupby(["domain", "stage", "ablation", "grade_mode"], dropna=False)["ap"].agg(["mean", "std", "count"])
    lines = ["| domain | stage | ablation | grade_mode | AP mean | AP std | n |", "|---|---|---|---|---:|---:|---:|"]
    for idx, row in g.iterrows():
        lines.append(
            f"| {idx[0]} | {idx[1]} | {idx[2]} | {idx[3]} | {row['mean']:.4f} | {row['std']:.4f} | {int(row['count'])} |"
        )
    return "\n".join(lines)


def architecture_figure() -> None:
    ensure()
    fig, ax = plt.subplots(figsize=(10, 5.2))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6)
    ax.axis("off")
    boxes = [
        (0.3, 4.2, "static\nResidualProjector", "#d6eaf8"),
        (0.3, 2.6, "aggregate\nResidualProjector", "#d5f5e3"),
        (0.3, 0.8, "temporal + mask\nLinear+LN", "#fdebd0"),
        (3.0, 4.2, "h_tab", "#d6eaf8"),
        (3.0, 2.2, "CNN 64 ch\nkernel 2 dil 1,2", "#f5b7b1"),
        (3.0, 0.6, "BiLSTM hid 128", "#d7bde2"),
        (5.6, 2.2, "softmax 3-way\nmasked gate", "#f9e79f"),
        (7.8, 2.2, "Head\nLN-128-GELU-1\np=σ(z)", "#aed6f1"),
    ]
    for x, y, t, c in boxes:
        ax.add_patch(mpatches.FancyBboxPatch((x, y), 2.0, 1.3, boxstyle="round,pad=0.05", facecolor=c, edgecolor="#2c3e50"))
        ax.text(x + 1.0, y + 0.65, t, ha="center", va="center", fontsize=8)
    ax.annotate("", xy=(3.0, 4.8), xytext=(2.3, 4.8), arrowprops=dict(arrowstyle="->"))
    ax.annotate("", xy=(3.0, 3.1), xytext=(2.3, 3.2), arrowprops=dict(arrowstyle="->"))
    ax.annotate("", xy=(3.0, 2.6), xytext=(2.3, 1.4), arrowprops=dict(arrowstyle="->"))
    ax.annotate("", xy=(3.0, 1.1), xytext=(2.3, 1.3), arrowprops=dict(arrowstyle="->"))
    ax.annotate("", xy=(5.6, 2.9), xytext=(5.0, 4.8), arrowprops=dict(arrowstyle="->"))
    ax.annotate("", xy=(5.6, 2.85), xytext=(5.0, 2.8), arrowprops=dict(arrowstyle="->"))
    ax.annotate("", xy=(5.6, 2.5), xytext=(5.0, 1.2), arrowprops=dict(arrowstyle="->"))
    ax.annotate("", xy=(7.8, 2.85), xytext=(7.6, 2.85), arrowprops=dict(arrowstyle="->"))
    ax.set_title("Hybrid CNN–BiLSTM — parallel CNN ∥ BiLSTM + tabular, masked softmax gate")
    fig.tight_layout()
    fig.savefig(FIG / "architecture_hybrid.png", dpi=160)
    plt.close(fig)


def write_ch1():
    text = """# Chương 1. Tổng quan đề tài nghiên cứu

## 1.1. Lý do chọn đề tài

Cảnh báo sớm nguy cơ học tập (early warning) cần mô hình xếp hạng được lớp thiểu số (trượt / rút) **trước** khi có điểm cuối kỳ. UCI Student Performance và OULAD là hai benchmark công khai, khác bản chất: UCI tĩnh T≤2, OULAD chuỗi VLE theo tuần.

## 1.2. Mục tiêu

- Xây dựng **một** kiến trúc Hybrid CNN–BiLSTM cho cả hai miền (khác chiều input và trọng số).
- Đánh giá bằng AP (`sklearn.metrics.average_precision_score`), group-split, FIT-only scale, không dùng outer test để chọn mô hình.
- Recommendation V xếp hành động khả thi trên `PredictionResult`, không ước lượng nhân quả.

## 1.3. Câu hỏi / giả thuyết kiểm định được

- **H1:** Trên UCI S1 và OULAD 35%, AP của Hybrid full **lớn hơn** AP tabular-only với Wilcoxon hai phía trên 9 run (α=0.05). → P0.1 ablation.
- **H2:** Trên UCI S1, AP Hybrid > AP LR và > AP RF (9 cặp fold×seed). → P0.2: **chấp nhận H2 trên S1** (p=0.0039 vs LR và vs RF). Trên OULAD 20% vs LR: **không** bác bỏ H0 (Δ≈0, p=0.57).
- **H3:** Cổng softmax tăng khối lượng CNN+BiLSTM khi cutoff tăng. → P0.5: OULAD tabular 0.315 (20%) → 0.172 (100%); BiLSTM 0.453 → 0.591.

## 1.4. Phạm vi

Không mở outer. Không dữ liệu sinh viên Việt Nam. Không user test. Recommendation V không phải ATE.
"""
    (CH / "CHUONG_1.md").write_text(text, encoding="utf-8")


def write_ch2():
    text = """# Chương 2. Cơ sở lý thuyết

## 2.1. Early warning và EDM

Bài toán xếp hạng nguy cơ nhị phân trên dữ liệu giáo dục: lệch lớp, rò rỉ thời gian, và tín hiệu tăng theo mốc quan sát. AP (average precision) phù hợp hơn ROC-AUC khi lớp dương là thiểu số và thứ tự xếp hạng quan trọng hơn ngưỡng mặc định 0.5.

Công thức AP dùng trong khóa luận: `sklearn.metrics.average_precision_score` — không phải diện tích thang PR hình thang tự viết.

## 2.2. UCI Student Performance và OULAD

- UCI (Cortez & Silva, 2008): bản ghi học kỳ, nhãn `G3 < 10`.
- OULAD (Kuzilek, Hlosta & Zdrahal, 2017): enrollment + VLE click, nhãn Fail|Withdrawn.

## 2.3. CNN–BiLSTM

CNN 1D trích mẫu cục bộ trên chuỗi có mask; BiLSTM mã hóa phụ thuộc hai chiều trong **cửa sổ đã cắt tại cutoff**. Fusion softmax 3 nhánh có mask tắt CNN/LSTM khi T=0.

Không dùng Transformer trong bản khóa vì T UCI = 2 (attention không có chuỗi dài) và cần so sánh công bằng với protocol đã chốt; đây là hướng phát triển, không phải kết luận Transformer kém.

## 2.4. Số liệu công bố (không cùng protocol — không so trực tiếp AP)

| Nguồn | Dữ liệu | Chỉ số | Ghi chú |
|---|---|---|---|
| Jha et al. 2019 | OULAD dropout / result | AUC tới ~0.91 / 0.93 (GBM, VLE) | Khác nhãn, khác split, ROC không AP |
| Kuznetsov 2025 | OULAD early warning | AUC 0.789 ngày 14; AP 0.722 | Ultra-early; GB ≈ LR |
| Frontiers 2026 BiLSTM+MLP+attention | OULAD | ROC-AUC 0.95; early weeks 0.91 | Khác protocol, có thể rò assessment |
| CNN–LSTM MDPI 2025 | OULAD | Accuracy 98.9% | Không cutoff-safe; không dùng làm trần |

Khóa luận **không** claim vượt 0.95 ROC. Số khóa là AP inner 3×3, Fail|Withdrawn, cutoff-safe.

## 2.5. Công thức dùng trong khóa luận

- BCE with logits, `pos_weight_FIT = (n_neg/n_pos)_FIT × hệ_số`.
- `p = σ(z)`; `ŷ = [p ≥ t]`; `t` chọn trên STOP (F1, rồi recall, rồi `|t−0.5|`).
- Bất định `H₂(p) = −[p log p + (1−p) log(1−p)] / log 2`.
- Cổng: `g = softmax(mask(W[h_tab;h_cnn;h_lstm;a;progress]))`, nhánh tắt logit = −∞.
- AP: average precision sklearn trên VALID.
"""
    (CH / "CHUONG_2.md").write_text(text, encoding="utf-8")


def write_ch3():
    text = f"""# Chương 3. Phân tích và thiết kế (bản research — bổ sung)

Giữ nguyên dàn ý khóa: dữ liệu, tensor, Hybrid, train, serving, Rec V. **Không đưa bảng AP vào chương này.**

Bổ sung sau P0/P1:

- Spearman **FIT-only** (3 fold) vs bản exploratory n=1044: xem `SPEARMAN_FIT.md`. G1/G2 vẫn ~−0.63/−0.67; absences vẫn yếu. Thứ tự thuộc tính không đổi.
- G1/G2 trên UCI serving code vẫn vào temporal **và** 5 cột aggregate. Ablation 3 arm (both / temporal_only / aggregate_only) định lượng trùng lặp — số ở Chương 4 research.
- Hình kiến trúc: `figures/architecture_hybrid.png` (thay ASCII).

Siêu tham số theo miền (lr, dropout, batch, pos_weight) lấy từ `configs/prediction/hybrid_final.json` / Phase3 HPO (`test_lab/.../uci_hpo_best.json`, `oulad_hpo_best.json`). Topology không đổi.

SMOTE: không chọn trên tensor (không tạo tuần VLE/G1 thật). Kết quả âm tính nằm log/research imbalance; không dùng focal/SMOTE trên bản khóa serving.
"""
    (CH / "CHUONG_3.md").write_text(text, encoding="utf-8")


def write_ch4():
    text = f"""# Chương 4. Kết quả thực nghiệm (bản research — bổ sung P0/P1)

Outer **không** dùng. AP = sklearn average_precision.

## 4.1. Kiểm định Hybrid vs LR/RF (9 run)

{_read("STAT_SIGNIFICANCE.md")}

**Diễn giải thẳng:** Hybrid **hơn** LR và RF có ý nghĩa trên UCI S1 và hầu hết mốc OULAD 35–100%. UCI S2 vs RF: ΔAP +0.003, CI chứa 0, p=0.16 — **không** tuyên bố hơn RF. OULAD 20% vs LR: hòa (Δ −0.0009, p=0.57). S0 Hybrid yếu hơn RF (CI âm) — đúng mốc CNN/BiLSTM tắt.

Số khóa UCI S1 là **0.8214**, không phải 0.811 (hard-code figure cũ, CSV tensor-parity không có hàng Hybrid).

## 4.2. Ablation module và G1/G2

{_ablation_table()}

Chi tiết: `ABLATION.md`.

## 4.3. Fail vs Withdrawn

{_read("LABEL_SPLIT_ANALYSIS.md")}

Withdrawn ngày càng ít ở cutoff muộn (2849 @20% → 496 @75% trong OOF). AP Withdrawn vs Pass ~0.55–0.61, thấp hơn Fail vs Pass. Gộp nhãn làm AP gộp trông cao hơn khả năng bắt Fail thuần.

## 4.4. Công bằng mô tả

{_read("FAIRNESS_BY_GROUP.md")}

Module AAA/GGG và IMD cao (70–90%) có AP thấp hơn overall ở 20–35%. Không sửa bias trong lượt này.

## 4.5. Cổng (XAI)

{_read("GATE_WEIGHTS.md")}

Hình: `figures/gate_weights_by_cutoff.png`. UCI S0 tabular=1. OULAD: mass BiLSTM tăng theo cutoff.

## 4.6. Đường cong và calibration

Hình OOF (join nhãn `final_result`, t = median STOP theo mốc): `pr_curves_oulad_oof.png`, `roc_curves_oulad_oof.png`, `confusion_oulad_oof.png`, `reliability_oulad_oof.png`.

## 4.7. Survivorship

{_read("SURVIVAL.md")}

4 184 enrollment có mặt ở 20% không còn ở 100%. Prevalence 0.424 → 0.317. AP 100% trên VALID ≈ AP trên giao với 20% vì bản 100% **chính là** tập còn lại. ΔAP 20→100 một phần đến từ mẫu dễ hơn, không chỉ “thêm tuần VLE”.

## 4.8. Spearman FIT-only

{_read("SPEARMAN_FIT.md")}
"""
    (CH / "CHUONG_4.md").write_text(text, encoding="utf-8")


def write_ch5():
    csv = ROOT / "artifacts" / "research" / "kltn_science_fix" / "ablation_raw.csv"
    abl = "đã có ablation_raw.csv" if csv.exists() else "TODO ablation đang chạy"
    text = f"""# Chương 5. Kết luận, hạn chế và hướng phát triển

## 5.1. Kết luận

- Hybrid CNN–BiLSTM xếp hạng nguy cơ trên UCI S1 (AP 0.8214) và OULAD 35–100% (0.806–0.920) theo protocol group-split, FIT-only, không outer.
- H2: hơn LR/RF có ý nghĩa trên S1 và hầu hết cutoff OULAD; **không** hơn LR tại 20%; **không** hơn RF tại S2.
- H3: cổng chuyển mass từ tabular sang BiLSTM khi chuỗi dài hơn.
- Recommendation V: NDCG@3 0.888 trên Panel C, không nhân quả; 57% INSUFFICIENT_EVIDENCE.
- Ablation: {abl}.

## 5.2. Hạn chế (nói thẳng)

1. Outer fold chưa mở — chưa có test cuối cùng.
2. 100% không phải early warning (prevalence giảm vì Withdrawn sớm bị loại; AP Withdrawn vs Pass thấp).
3. Rec V không phải can thiệp nhân quả; gold Panel C phụ thuộc LLM.
4. S0/20% không phải claim chính của CNN–BiLSTM (T=0 hoặc tín hiệu yếu; 20% hòa LR).
5. UCI T=2: CNN dilation 2 gần như thoái hóa; ΔAP S0→S1 chủ yếu là G1 xuất hiện (ablation sẽ định lượng).
6. Không dữ liệu / GV Việt Nam; serving đọc OOF, không forward clickstream live; không FastAPI.
7. Không phân tích XAI điểm (SHAP); chỉ mass cổng trung bình.
8. G1/G2 vẫn nhân đôi trên pipeline serving cho đến khi user duyệt bỏ arm `both`.
9. Hai số AP 0.821 vs 0.811: 0.821 là khóa; 0.811 là hard-code figure tensor-parity thiếu hàng Hybrid.

## 5.3. Hướng phát triển

- Mở outer **một lần** sau khi đóng băng mọi lựa chọn (cần user duyệt).
- Tách mô hình Fail vs Withdrawn.
- Dữ liệu trường VN + user test GV.
- Live forward từ VLE; API; đo latency.
- Transformer / attention trên OULAD tuần dài.
- Bỏ trùng G1/G2 nếu ablation temporal_only không kém `both`.
- Hiệu chỉnh xác suất (temperature / isotonic) vì ECE S0 còn cao.
"""
    (CH / "CHUONG_5.md").write_text(text, encoding="utf-8")


def write_summary():
    csv = ROOT / "artifacts" / "research" / "kltn_science_fix" / "ablation_raw.csv"
    n = len(pd.read_csv(csv)) if csv.exists() else 0
    text = f"""# SUMMARY — kltn science fix (research)

| Mục | Trạng thái | Output |
|---|---|---|
| P0.1 Ablation | {'đang/đã có '+str(n)+' hàng' if n else 'TODO GPU'} | `ABLATION.md`, `artifacts/research/kltn_science_fix/ablation_raw.csv` |
| P0.2 Wilcoxon + dual AP | XONG | `STAT_SIGNIFICANCE.md` |
| P0.3 Fail/Withdrawn | XONG | `LABEL_SPLIT_ANALYSIS.md` |
| P0.4 Fairness | XONG | `FAIRNESS_BY_GROUP.md`, `figures/fairness_ap_by_group_35pct.png` |
| P0.5 Gate | XONG | `GATE_WEIGHTS.md`, `figures/gate_weights_by_cutoff.png` |
| P0.6 Ch5 | XONG bản research | `chapters/CHUONG_5.md` |
| P1 PR/ROC/CM/reliability | XONG | `figures/pr_*.png` `roc_*` `confusion_*` `reliability_*` |
| P1 Spearman FIT | XONG | `SPEARMAN_FIT.md` |
| P1 Survivorship | XONG | `SURVIVAL.md` |
| P2 Ch1 Ch2 | XONG bản research | `chapters/CHUONG_1.md` `CHUONG_2.md` |
| Ch3/Ch4 bổ sung | XONG bản research | `chapters/CHUONG_3.md` `CHUONG_4.md` |
| Hình kiến trúc | XONG | `figures/architecture_hybrid.png` |
| Outer test | **Không mở** | — |
| FastAPI / user test | Không làm | Ch5 hướng phát triển |

## Mâu thuẫn số liệu đã giải quyết

Serving UCI S1 AP **0.8214** = ROBUST_CONFIRMATION L1. Số **0.811** không có trong CSV tensor-parity (thiếu hàng Hybrid), chỉ hard-code trong `generate_ch4_figures.py`.

## Mâu thuẫn / việc chưa merge main

Toàn bộ nằm `reports/research/hybrid_superiority_v2/` và `research/kltn_science_fix/`. User duyệt rồi mới đưa vào Chương nộp.
"""
    (REP / "SUMMARY.md").write_text(text, encoding="utf-8")


def write_notebook():
    nb = {
        "nbformat": 4,
        "nbformat_minor": 5,
        "metadata": {"kernelspec": {"display_name": "Python 3.10", "language": "python", "name": "python3"}},
        "cells": [
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "# Research figures — Hybrid KLTN science fix\n",
                    "Chỉ vẽ từ CSV/PNG đã khóa hoặc ablation research. Không train trong notebook này.\n",
                ],
            },
            {
                "cell_type": "code",
                "metadata": {},
                "outputs": [],
                "execution_count": None,
                "source": [
                    "from pathlib import Path\n",
                    "from IPython.display import Image, display, Markdown\n",
                    "import pandas as pd\n",
                    "FIG = Path('reports/research/hybrid_superiority_v2/figures')\n",
                    "REP = Path('reports/research/hybrid_superiority_v2')\n",
                    "for p in sorted(FIG.glob('*.png')):\n",
                    "    display(Markdown(f'**{p.name}**'))\n",
                    "    display(Image(filename=str(p), width=720))\n",
                    "csv = Path('artifacts/research/kltn_science_fix/ablation_raw.csv')\n",
                    "if csv.exists():\n",
                    "    df = pd.read_csv(csv)\n",
                    "    display(df.groupby(['domain','stage','ablation'])['ap'].mean().unstack().round(4))\n",
                ],
            },
        ],
    }
    path = REP / "CHUONG_4_RESEARCH.ipynb"
    path.write_text(json.dumps(nb, indent=2), encoding="utf-8")


def main() -> None:
    ensure()
    architecture_figure()
    write_ch1()
    write_ch2()
    write_ch3()
    write_ch4()
    write_ch5()
    write_summary()
    write_notebook()
    print("wrote chapters, SUMMARY, notebook, architecture figure")


if __name__ == "__main__":
    main()
