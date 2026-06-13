from __future__ import annotations

import json
import math
import re
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont


ROOT = Path(r"C:\Huflit\kltn")
OUT = ROOT / "reports" / "final" / "figures" / "thesis_revised"
OUT.mkdir(parents=True, exist_ok=True)

DATASETS = ["student-mat", "student-por", "xapi"]
LABELS = ["Student-Mat", "Student-Por", "xAPI"]
CLASSES = ["Low", "Medium", "High"]

WHITE = "#FFFFFF"
BLACK = "#222222"
GRAY = "#666666"
LIGHT = "#E6E6E6"
GRID = "#D7D7D7"
BLUE = "#4C78A8"
MID_BLUE = "#8FB3D9"
ORANGE = "#F2A541"
GREEN = "#6B9E78"
RED = "#C65D57"
SERIES = [BLUE, ORANGE, GREEN, RED]


def font(size: int, bold: bool = False):
    folder = Path(r"C:\Windows\Fonts")
    path = folder / ("arialbd.ttf" if bold else "arial.ttf")
    return ImageFont.truetype(str(path), size)


def image(width=1800, height=1000):
    value = Image.new("RGB", (width, height), WHITE)
    return value, ImageDraw.Draw(value)


def centered(draw, box, text, size=28, bold=False, fill=BLACK):
    x1, y1, x2, y2 = box
    fnt = font(size, bold)
    lines = str(text).split("\n")
    heights = [draw.textbbox((0, 0), line, font=fnt)[3] for line in lines]
    y = y1 + ((y2 - y1) - sum(heights) - 5 * (len(lines) - 1)) / 2
    for line, line_height in zip(lines, heights):
        bounds = draw.textbbox((0, 0), line, font=fnt)
        x = x1 + ((x2 - x1) - (bounds[2] - bounds[0])) / 2
        draw.text((x, y), line, font=fnt, fill=fill)
        y += line_height + 5


def title(draw, text, subtitle=None, width=1800):
    centered(draw, (50, 22, width - 50, 75), text, 34, True)
    if subtitle:
        centered(draw, (70, 75, width - 70, 115), subtitle, 20, False, GRAY)


def save(value, filename):
    value.save(OUT / filename, dpi=(200, 200), optimize=True)


def box(draw, xy, heading, body="", fill=WHITE):
    draw.rectangle(xy, fill=fill, outline=BLACK, width=2)
    x1, y1, x2, y2 = xy
    centered(draw, (x1 + 8, y1 + 8, x2 - 8, y1 + 58), heading, 23, True)
    if body:
        centered(draw, (x1 + 12, y1 + 62, x2 - 12, y2 - 10), body, 19)


def arrow(draw, start, end):
    draw.line((start, end), fill=BLACK, width=3)
    angle = math.atan2(end[1] - start[1], end[0] - start[0])
    length = 14
    points = [
        end,
        (end[0] - length * math.cos(angle - 0.55), end[1] - length * math.sin(angle - 0.55)),
        (end[0] - length * math.cos(angle + 0.55), end[1] - length * math.sin(angle + 0.55)),
    ]
    draw.polygon(points, fill=BLACK)


def axes(draw, plot, ymin, ymax, ticks=5, percent=False):
    x1, y1, x2, y2 = plot
    draw.line((x1, y1, x1, y2), fill=BLACK, width=2)
    draw.line((x1, y2, x2, y2), fill=BLACK, width=2)
    for idx in range(ticks + 1):
        value = ymin + (ymax - ymin) * idx / ticks
        y = y2 - (y2 - y1) * idx / ticks
        draw.line((x1, y, x2, y), fill=GRID, width=1)
        label = f"{value * 100:.0f}%" if percent else (f"{value:.2f}" if ymax <= 1.2 else f"{value:.0f}")
        bounds = draw.textbbox((0, 0), label, font=font(18))
        draw.text((x1 - (bounds[2] - bounds[0]) - 10, y - 10), label, font=font(18), fill=GRAY)


def grouped_bars(filename, chart_title, groups, names, values, ymin=0, ymax=None, subtitle=None):
    value, draw = image()
    title(draw, chart_title, subtitle)
    matrix = np.asarray(values, dtype=float)
    ymax = float(ymax or matrix.max() * 1.15)
    plot = (135, 145, 1745, 830)
    axes(draw, plot, ymin, ymax)
    x1, _, x2, y2 = plot
    group_width = (x2 - x1) / len(groups)
    bar_width = min(76, group_width / (len(names) + 1))
    for group_idx, group in enumerate(groups):
        center_x = x1 + group_width * (group_idx + 0.5)
        centered(draw, (center_x - group_width / 2, y2 + 10, center_x + group_width / 2, y2 + 52), group, 20, True)
        for series_idx, name in enumerate(names):
            metric = float(matrix[series_idx, group_idx])
            left = center_x - len(names) * bar_width / 2 + series_idx * bar_width
            top = y2 - (metric - ymin) / (ymax - ymin) * (y2 - plot[1])
            draw.rectangle((left, top, left + bar_width * 0.75, y2), fill=SERIES[series_idx], outline=BLACK, width=1)
            label = f"{metric:.3f}" if ymax <= 1.2 else str(int(metric))
            centered(draw, (left - 12, top - 30, left + bar_width + 12, top - 3), label, 16, True)
    legend_x = 360
    for idx, name in enumerate(names):
        x = legend_x + idx * 330
        draw.rectangle((x, 915, x + 28, 943), fill=SERIES[idx], outline=BLACK)
        draw.text((x + 40, 914), name, font=font(19), fill=BLACK)
    save(value, filename)


def load_results():
    results = {}
    for dataset in DATASETS:
        metrics = json.loads((ROOT / "reports" / "final" / "metrics" / f"{dataset}_3class_locked_test_metrics.json").read_text(encoding="utf-8"))
        predictions = pd.read_csv(ROOT / "reports" / "final" / "predictions" / f"{dataset}_3class_predictions.csv")
        recommendations = pd.read_csv(ROOT / "reports" / "final" / "recommendations" / f"{dataset}_3class_learning_paths.csv")
        importance = pd.read_csv(ROOT / "reports" / "final" / "explanations" / f"{dataset}_3class_feature_importance.csv")
        report = (ROOT / "reports" / "final" / f"{dataset}_3class_final_report.txt").read_text(encoding="utf-8")
        match = re.search(r"Optuna Best CV F1:\s*([0-9.]+)", report)
        cv = float(match.group(1)) if match else 0.0
        cm = pd.crosstab(predictions.True_Label, predictions.Pred_Label).reindex(index=[0, 1, 2], columns=[0, 1, 2], fill_value=0).to_numpy()
        per_class = []
        for cls in range(3):
            tp = cm[cls, cls]
            fp = cm[:, cls].sum() - tp
            fn = cm[cls, :].sum() - tp
            precision = tp / (tp + fp) if tp + fp else 0
            recall = tp / (tp + fn) if tp + fn else 0
            f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0
            per_class.append((precision, recall, f1, int(cm[cls].sum())))
        results[dataset] = {
            "metrics": metrics,
            "predictions": predictions,
            "recommendations": recommendations,
            "importance": importance,
            "cv": cv,
            "cm": cm,
            "per_class": per_class,
        }
    if results["xapi"]["cv"] <= 0:
        db = ROOT / "models" / "saved" / "final" / "xapi_3class_optuna.db"
        with sqlite3.connect(db) as connection:
            results["xapi"]["cv"] = float(connection.execute("SELECT MAX(value) FROM trial_values").fetchone()[0])
    return results


RESULTS = load_results()


def diagrams():
    value, draw = image(1800, 900)
    title(draw, "Quy trình thực nghiệm", "Locked test được giữ riêng cho bước đánh giá cuối")
    nodes = [
        (60, "Dữ liệu", "Student-Mat\nStudent-Por\nxAPI"),
        (385, "Phân hoạch", "Train pool 80%\nLocked test 20%"),
        (710, "Tối ưu", "Repeated stratified\n5-fold × 3"),
        (1035, "Huấn luyện", "Tách validation trước\nresampling theo seed"),
        (1360, "Đánh giá", "F1-Macro, Accuracy\nPrecision, Recall\nRMSE, R²"),
    ]
    for x, heading, body in nodes:
        box(draw, (x, 250, x + 270, 485), heading, body)
    for x, _, _ in nodes[:-1]:
        arrow(draw, (x + 270, 368), (x + 320, 368))
    centered(draw, (140, 610, 1660, 750), "Dự đoán → phân tích rủi ro theo luật → lộ trình học tập → lưu PostgreSQL", 26, True)
    save(value, "01_system_pipeline.png")

    value, draw = image(1800, 980)
    title(draw, "Kiến trúc CNN-BiLSTM + Context MLP")
    box(draw, (65, 180, 330, 340), "Đầu vào tuần tự", "G1-G2 hoặc 4 chỉ báo xAPI")
    box(draw, (450, 170, 775, 350), "CNN-BiLSTM", "Conv1D → BiLSTM\n→ Attention pooling")
    box(draw, (65, 590, 330, 750), "Đầu vào bối cảnh", "Biến số và biến phân loại")
    box(draw, (450, 555, 775, 785), "Context MLP", "Min-Max cho biến số\nEmbedding cho biến phân loại\nMLP hai tầng")
    box(draw, (960, 345, 1225, 590), "Fusion", "Ghép hai vector\nLinear → ReLU\nDropout")
    box(draw, (1390, 200, 1710, 400), "Student-Mat / Por", "3 logits\nSoftmax ba lớp")
    box(draw, (1390, 565, 1710, 785), "xAPI ordinal", "2 logits ngưỡng\nBCEWithLogitsLoss\nquy đổi về 3 xác suất")
    arrow(draw, (330, 260), (450, 260)); arrow(draw, (330, 670), (450, 670))
    arrow(draw, (775, 260), (960, 420)); arrow(draw, (775, 670), (960, 520))
    arrow(draw, (1225, 425), (1390, 300)); arrow(draw, (1225, 520), (1390, 675))
    save(value, "02_hybrid_architecture.png")

    value, draw = image(1800, 900)
    title(draw, "Kiểm soát rò rỉ dữ liệu")
    box(draw, (80, 180, 390, 350), "Dữ liệu gốc", "Tách một lần với seed 42")
    box(draw, (570, 130, 970, 340), "Train pool", "Optuna và ensemble")
    box(draw, (1240, 130, 1660, 340), "Locked test", "Không dùng để chọn tham số\nhoặc epoch")
    arrow(draw, (390, 260), (570, 235)); arrow(draw, (390, 260), (1240, 235))
    box(draw, (270, 560, 650, 770), "Train fold / train seed", "Fit encoder, scaler\nResampling\nFit feature selector")
    box(draw, (850, 560, 1230, 770), "Validation", "Tách trước resampling\nChỉ transform")
    arrow(draw, (770, 340), (460, 560)); arrow(draw, (770, 340), (1040, 560))
    save(value, "03_leakage_control.png")

    value, draw = image(1800, 900)
    title(draw, "Quy trình sinh lộ trình học tập")
    nodes = [
        (80, "Kết quả mô hình", "Lớp dự đoán\nvà confidence"),
        (420, "Thông tin người học", "Điểm, chuyên cần,\ntương tác, hỗ trợ"),
        (760, "Luật rủi ro", "Điều kiện và\nmức ưu tiên"),
        (1100, "Mức rủi ro", "stable\nmoderate\nhigh"),
        (1440, "Lộ trình", "Mục tiêu và hành động\ntheo tuần"),
    ]
    for x, heading, body in nodes:
        box(draw, (x, 260, x + 270, 500), heading, body)
    for x, _, _ in nodes[:-1]:
        arrow(draw, (x + 270, 380), (x + 330, 380))
    save(value, "04_rule_engine.png")

    value, draw = image(1800, 980)
    title(draw, "Các bảng PostgreSQL dùng để lưu kết quả")
    box(draw, (650, 145, 1150, 315), "paper_runs", "Thông tin một lần chạy")
    box(draw, (70, 550, 500, 790), "paper_predictions", "Nhãn thật và dự đoán\nXác suất, confidence\nĐặc trưng gốc")
    box(draw, (685, 550, 1115, 790), "paper_evaluation_metrics", "Accuracy, Precision\nRecall, F1-Macro\nRMSE, R²")
    box(draw, (1300, 550, 1730, 790), "paper_learning_recommendations", "Risk band\nFeature snapshot\nLearning path")
    arrow(draw, (790, 315), (290, 550)); arrow(draw, (900, 315), (900, 550)); arrow(draw, (1010, 315), (1515, 550))
    save(value, "05_postgresql_schema.png")


def empirical_figures():
    counts = []
    for dataset in DATASETS:
        frame = RESULTS[dataset]["predictions"]
        counts.append([int((frame.True_Label == cls).sum()) for cls in range(3)])
    grouped_bars("06_class_distribution.png", "Phân bố lớp trong locked test", LABELS, CLASSES, np.asarray(counts).T, ymax=95)

    metric_names = ["Accuracy", "Precision", "Recall", "F1-Macro"]
    metric_keys = ["Accuracy", "Precision-Macro", "Recall-Macro", "F1-Macro"]
    metric_values = [[RESULTS[d]["metrics"][key] for d in DATASETS] for key in metric_keys]
    grouped_bars("07_locked_metrics.png", "Kết quả trên locked test", LABELS, metric_names, metric_values, ymin=0.70, ymax=0.94)

    value, draw = image()
    title(draw, "F1-Macro: cross-validation và locked test")
    plot = (170, 150, 1690, 830)
    axes(draw, plot, 0.75, 0.94)
    for idx, dataset in enumerate(DATASETS):
        x = 420 + idx * 500
        cv = RESULTS[dataset]["cv"]
        test = RESULTS[dataset]["metrics"]["F1-Macro"]
        y_cv = plot[3] - (cv - 0.75) / (0.94 - 0.75) * (plot[3] - plot[1])
        y_test = plot[3] - (test - 0.75) / (0.94 - 0.75) * (plot[3] - plot[1])
        draw.line((x, y_cv, x, y_test), fill=GRAY, width=3)
        draw.ellipse((x - 12, y_cv - 12, x + 12, y_cv + 12), fill=BLUE, outline=BLACK)
        draw.ellipse((x - 12, y_test - 12, x + 12, y_test + 12), fill=ORANGE, outline=BLACK)
        draw.text((x + 22, y_cv - 14), f"{cv:.3f}", font=font(18), fill=BLACK)
        draw.text((x + 22, y_test - 14), f"{test:.3f}", font=font(18), fill=BLACK)
        centered(draw, (x - 150, 845, x + 150, 890), LABELS[idx], 21, True)
    draw.ellipse((640, 925, 660, 945), fill=BLUE, outline=BLACK); draw.text((675, 922), "CV tốt nhất", font=font(18), fill=BLACK)
    draw.ellipse((960, 925, 980, 945), fill=ORANGE, outline=BLACK); draw.text((995, 922), "Locked test", font=font(18), fill=BLACK)
    save(value, "08_cv_test_gap.png")

    value, draw = image(1800, 940)
    title(draw, "Ma trận nhầm lẫn trên locked test", "Hàng: nhãn thật; cột: nhãn dự đoán")
    for panel, dataset in enumerate(DATASETS):
        cm = RESULTS[dataset]["cm"]
        ox = 100 + panel * 575
        centered(draw, (ox, 130, ox + 500, 180), LABELS[panel], 25, True)
        max_value = max(1, int(cm.max()))
        for col, label in enumerate(CLASSES):
            centered(draw, (ox + 110 + col * 125, 190, ox + 235 + col * 125, 230), label, 18, True)
        for row, label in enumerate(CLASSES):
            centered(draw, (ox, 245 + row * 125, ox + 100, 370 + row * 125), label, 18, True)
            for col in range(3):
                count = int(cm[row, col])
                shade = int(245 - 120 * count / max_value)
                cell = (ox + 110 + col * 125, 245 + row * 125, ox + 235 + col * 125, 370 + row * 125)
                draw.rectangle(cell, fill=(shade, shade, shade), outline=WHITE, width=3)
                centered(draw, cell, str(count), 28, True)
    save(value, "09_confusion_matrices.png")

    f1_values = [[RESULTS[d]["per_class"][cls][2] for d in DATASETS] for cls in range(3)]
    grouped_bars("10_per_class_f1.png", "F1-score theo lớp", LABELS, CLASSES, f1_values, ymin=0.60, ymax=1.0)

    value, draw = image(1800, 1060)
    title(draw, "Permutation feature importance", "Kết quả hiện tại có nhiều giá trị bằng 0; không dùng để xếp hạng nguyên nhân")
    for panel, dataset in enumerate(DATASETS):
        frame = RESULTS[dataset]["importance"].sort_values("Importance", ascending=False).head(8).iloc[::-1]
        ox = 60 + panel * 590
        centered(draw, (ox, 130, ox + 540, 180), LABELS[panel], 25, True)
        maximum = max(float(frame.Importance.clip(lower=0).max()), 0.001)
        for idx, row in enumerate(frame.itertuples()):
            y = 205 + idx * 92
            centered(draw, (ox, y, ox + 230, y + 55), str(row.Feature), 16)
            bar_length = 260 * max(float(row.Importance), 0) / maximum
            draw.rectangle((ox + 235, y + 17, ox + 235 + bar_length, y + 43), fill=BLUE)
            draw.text((ox + 245 + bar_length, y + 17), f"{row.Importance:.3f}", font=font(15), fill=GRAY)
    save(value, "11_feature_importance.png")

    bins = [0, .5, .6, .7, .8, .9, 1.001]
    bin_labels = ["<0.5", "0.5-0.6", "0.6-0.7", "0.7-0.8", "0.8-0.9", "≥0.9"]
    hist_values = []
    for dataset in DATASETS:
        counts = pd.cut(RESULTS[dataset]["predictions"].Confidence, bins=bins, right=False).value_counts().sort_index().tolist()
        hist_values.append(counts)
    grouped_bars("12_confidence_distribution.png", "Phân bố confidence", bin_labels, LABELS, hist_values, ymax=max(max(row) for row in hist_values) * 1.15)

    risk_names = ["stable", "moderate", "high"]
    risk_values = []
    for dataset in DATASETS:
        counts = RESULTS[dataset]["recommendations"].risk_band.value_counts()
        risk_values.append([int(counts.get(name, 0)) for name in risk_names])
    grouped_bars("13_risk_band_distribution.png", "Phân bố mức rủi ro", LABELS, risk_names, np.asarray(risk_values).T, ymax=80)


def main():
    diagrams()
    empirical_figures()
    print(f"Generated {len(list(OUT.glob('*.png')))} figures in {OUT}")


if __name__ == "__main__":
    main()
