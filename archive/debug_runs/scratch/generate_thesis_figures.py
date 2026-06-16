from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from PIL import Image, ImageDraw, ImageFont


ROOT = Path(r"C:\Huflit\kltn")
OUT = ROOT / "reports" / "final" / "figures" / "thesis"
OUT.mkdir(parents=True, exist_ok=True)

NAVY = "#17365D"
BLUE = "#4472C4"
SKY = "#D9EAF7"
PALE = "#F3F6FA"
GREEN = "#70AD47"
PALE_GREEN = "#E2F0D9"
GOLD = "#FFC000"
PALE_GOLD = "#FFF2CC"
RED = "#C00000"
GRAY = "#666666"
GRID = "#D9E1F2"
WHITE = "#FFFFFF"
BLACK = "#111111"


def font(size: int, bold: bool = False):
    name = "timesbd.ttf" if bold else "times.ttf"
    return ImageFont.truetype(str(Path(r"C:\Windows\Fonts") / name), size)


def canvas(w=1800, h=1050):
    return Image.new("RGB", (w, h), WHITE), ImageDraw.Draw(Image.new("RGB", (1, 1)))


def new_image(w=1800, h=1050):
    im = Image.new("RGB", (w, h), WHITE)
    return im, ImageDraw.Draw(im)


def text_center(draw, box, text, size=38, bold=False, fill=BLACK, spacing=8):
    x1, y1, x2, y2 = box
    f = font(size, bold)
    lines = text.split("\n")
    heights = [draw.textbbox((0, 0), line, font=f)[3] for line in lines]
    total = sum(heights) + spacing * (len(lines) - 1)
    y = y1 + (y2 - y1 - total) / 2
    for line, height in zip(lines, heights):
        bbox = draw.textbbox((0, 0), line, font=f)
        x = x1 + (x2 - x1 - (bbox[2] - bbox[0])) / 2
        draw.text((x, y), line, font=f, fill=fill)
        y += height + spacing


def box(draw, xy, title, body="", fill=SKY, outline=NAVY, title_size=36, body_size=28):
    draw.rounded_rectangle(xy, radius=18, fill=fill, outline=outline, width=4)
    x1, y1, x2, y2 = xy
    text_center(draw, (x1 + 12, y1 + 8, x2 - 12, y1 + 72), title, title_size, True, NAVY)
    if body:
        text_center(draw, (x1 + 18, y1 + 76, x2 - 18, y2 - 10), body, body_size, False, BLACK)


def arrow(draw, start, end, color=NAVY, width=5):
    draw.line([start, end], fill=color, width=width)
    import math
    angle = math.atan2(end[1] - start[1], end[0] - start[0])
    length = 20
    left = (end[0] - length * math.cos(angle - 0.55), end[1] - length * math.sin(angle - 0.55))
    right = (end[0] - length * math.cos(angle + 0.55), end[1] - length * math.sin(angle + 0.55))
    draw.polygon([end, left, right], fill=color)


def title(draw, text, subtitle=None):
    text_center(draw, (40, 18, 1760, 90), text, 46, True, NAVY)
    if subtitle:
        text_center(draw, (40, 82, 1760, 125), subtitle, 25, False, GRAY)


def save(im, name):
    path = OUT / name
    im.save(path, dpi=(220, 220))
    return path


def pipeline():
    im, d = new_image(1900, 1120)
    title(d, "Quy trình hệ thống dự đoán và khuyến nghị", "Locked test được cô lập trước mọi hoạt động tối ưu")
    xs = [55, 385, 715, 1045, 1375]
    labels = [
        ("Dữ liệu đầu vào", "Student-Mat\nStudent-Por\nxAPI-Edu-Data"),
        ("Tách dữ liệu", "Train pool 80%\nLocked test 20%\nStratified, seed 42"),
        ("CV + tiền xử lý", "Feature engineering\nScale/encode\nSMOTE/ADASYN\nFeature selection"),
        ("Tối ưu & huấn luyện", "Optuna 5-fold CV\n50/50/150 trials\nEnsemble 5 seeds"),
        ("Đánh giá", "Accuracy, F1-Macro\nPrecision, Recall\nRMSE, R²"),
    ]
    for x, (h, b) in zip(xs, labels):
        box(d, (x, 180, x + 280, 420), h, b, SKY, title_size=31)
    for x in xs[:-1]:
        arrow(d, (x + 280, 300), (x + 325, 300))
    box(d, (210, 610, 630, 865), "Giải thích mô hình", "Permutation importance\nXác suất và confidence", PALE)
    box(d, (740, 610, 1160, 865), "Learning Path Engine", "Risk factors theo luật\nKế hoạch can thiệp 1-4 tuần", PALE_GOLD)
    box(d, (1270, 610, 1690, 865), "PostgreSQL", "Runs • Predictions\nMetrics • Recommendations", PALE_GREEN)
    arrow(d, (1515, 420), (1480, 610))
    arrow(d, (1270, 738), (1160, 738))
    arrow(d, (740, 738), (630, 738))
    text_center(d, (170, 930, 1730, 1060), "Đầu ra có thể truy vết: nhãn dự đoán, vector xác suất, độ tin cậy, yếu tố rủi ro và lộ trình học tập", 31, True, NAVY)
    save(im, "01_system_pipeline.png")


def architecture():
    im, d = new_image(1900, 1120)
    title(d, "Kiến trúc lai CNN-BiLSTM + Context MLP", "Hai nhánh xử lý dữ liệu tuần tự và dữ liệu bối cảnh trước khi hợp nhất")
    box(d, (60, 170, 330, 370), "Đầu vào tuần tự", "Student: G1, G2\nxAPI: 4 chỉ báo", PALE)
    box(d, (450, 145, 830, 395), "Nhánh CNN-BiLSTM", "Conv1D + BatchNorm\nReLU + Dropout\nBiLSTM hai chiều", SKY)
    box(d, (930, 165, 1260, 375), "Attention Pooling", "Trọng số softmax\nTổng có trọng số", SKY)
    box(d, (60, 655, 330, 855), "Đầu vào bối cảnh", "Biến số\nBiến phân loại", PALE)
    box(d, (450, 625, 830, 885), "Context MLP", "Min-Max scaling\nLabel encoding\nLinear-ReLU-Dropout\nLinear-ReLU", SKY)
    box(d, (1370, 390, 1685, 670), "Fusion", "Concatenate\nLinear + ReLU\nDropout", PALE_GOLD)
    box(d, (1715, 420, 1885, 660), "Đầu ra", "3 logits\nSoftmax\nLow\nMedium\nHigh", PALE_GREEN, title_size=25, body_size=20)
    arrow(d, (330, 270), (450, 270)); arrow(d, (830, 270), (930, 270))
    arrow(d, (330, 755), (450, 755)); arrow(d, (1260, 270), (1370, 475)); arrow(d, (830, 755), (1370, 590)); arrow(d, (1685, 530), (1730, 530))
    text_center(d, (380, 955, 1520, 1055), "Ensemble: trung bình hóa xác suất của 5 mô hình được huấn luyện với các seed cố định", 31, True, NAVY)
    save(im, "02_hybrid_architecture.png")


def leakage_control():
    im, d = new_image(1800, 1050)
    title(d, "Giao thức chống rò rỉ dữ liệu", "Mọi trạng thái tiền xử lý được fit trên phần train tương ứng")
    box(d, (70, 180, 420, 365), "Dữ liệu gốc", "Tách stratified một lần", PALE)
    box(d, (560, 120, 970, 350), "Train pool 80%", "Tham gia Optuna\nvà huấn luyện ensemble", SKY)
    box(d, (1210, 120, 1640, 350), "Locked test 20%", "Không dùng chọn feature,\nsiêu tham số hoặc epoch", PALE_GOLD)
    arrow(d, (420, 270), (560, 235)); arrow(d, (420, 270), (1210, 235))
    box(d, (150, 570, 500, 820), "Train fold", "Fit scaler/encoder\nResampling\nFit feature selector", SKY)
    box(d, (725, 570, 1075, 820), "Validation fold", "Chỉ transform\nKhông oversampling", PALE_GREEN)
    box(d, (1300, 570, 1650, 820), "Điểm trial", "F1-Macro của fold\nTrung bình 5 fold", PALE)
    arrow(d, (765, 350), (380, 570)); arrow(d, (765, 350), (900, 570)); arrow(d, (500, 695), (725, 695)); arrow(d, (1075, 695), (1300, 695))
    text_center(d, (120, 900, 1680, 1005), "Nguyên tắc kiểm định: locked test chỉ được mở sau khi kiến trúc và siêu tham số đã cố định", 31, True, RED)
    save(im, "03_leakage_control.png")


def rule_engine():
    im, d = new_image(1800, 1050)
    title(d, "Luồng xử lý của Rule-based Learning Path Engine")
    nodes = [
        (70, "Dự đoán", "Lớp + confidence"), (405, "Feature snapshot", "Điểm, chuyên cần,\ntương tác, hỗ trợ"),
        (740, "Phát hiện rủi ro", "Luật ngưỡng\npriority 1-3"), (1075, "Xếp risk band", "stable / moderate / high"),
        (1410, "Sinh lộ trình", "Mục tiêu và hành động\ntheo tuần"),
    ]
    for x, h, b in nodes:
        box(d, (x, 250, x + 280, 500), h, b, SKY if x < 1000 else PALE_GOLD)
    for x, *_ in nodes[:-1]: arrow(d, (x + 280, 375), (x + 325, 375))
    phases = [(140, "Tuần 1", "Khôi phục chuyên cần"), (530, "Tuần 1-2", "Bù kiến thức / học liệu"), (920, "Tuần 2-4", "Ổn định tự học / tương tác"), (1310, "Cuối tuần", "Đánh giá và điều chỉnh")]
    for x, h, b in phases: box(d, (x, 700, x + 350, 885), h, b, PALE_GREEN, title_size=31, body_size=25)
    arrow(d, (1550, 500), (1485, 700))
    save(im, "04_rule_engine.png")


def database_schema():
    im, d = new_image(1800, 1100)
    title(d, "Mô hình lưu trữ PostgreSQL")
    box(d, (650, 120, 1150, 330), "paper_runs", "paper_run_id (PK)\ngenerated_at • status\nrun_payload JSONB", SKY)
    box(d, (70, 500, 500, 810), "paper_predictions", "run_id (FK)\ndataset • split • row_index\ntrue/predicted label\nprobability • confidence\noriginal_features JSONB", PALE)
    box(d, (685, 500, 1115, 810), "paper_evaluation_metrics", "run_id (FK)\naccuracy • precision_macro\nrecall_macro • f1_macro\nrmse • r2 • payload", PALE_GREEN)
    box(d, (1300, 500, 1730, 810), "paper_learning_recommendations", "run_id (FK)\nrisk_band • confidence\nfeature_snapshot JSONB\nrecommended_learning_path JSONB", PALE_GOLD, title_size=27, body_size=24)
    arrow(d, (780, 330), (300, 500)); arrow(d, (900, 330), (900, 500)); arrow(d, (1020, 330), (1515, 500))
    text_center(d, (180, 900, 1620, 1020), "Mỗi phiên chạy liên kết dự đoán, metric và khuyến nghị để hỗ trợ kiểm toán và tái lập", 33, True, NAVY)
    save(im, "05_postgresql_schema.png")


def axes(draw, box_xy, ymin, ymax, ticks=5, ylabel=None):
    x1, y1, x2, y2 = box_xy
    draw.line((x1, y1, x1, y2), fill=BLACK, width=3); draw.line((x1, y2, x2, y2), fill=BLACK, width=3)
    for i in range(ticks + 1):
        val = ymin + (ymax - ymin) * i / ticks
        y = y2 - (y2 - y1) * i / ticks
        draw.line((x1, y, x2, y), fill=GRID, width=2)
        draw.text((x1 - 75, y - 14), f"{val:.2f}", font=font(24), fill=GRAY)
    if ylabel: draw.text((20, (y1 + y2) / 2), ylabel, font=font(26, True), fill=NAVY)


def bar_chart(name, chart_title, groups, series, values, ymin=0, ymax=1):
    im, d = new_image(1800, 1050); title(d, chart_title)
    plot = (150, 150, 1710, 860); axes(d, plot, ymin, ymax, 5)
    x1, y1, x2, y2 = plot; colors = [BLUE, GREEN, GOLD, RED]
    group_w = (x2 - x1) / len(groups); bar_w = group_w / (len(series) + 1)
    for gi, group in enumerate(groups):
        center = x1 + group_w * (gi + .5)
        d.text((center - 80, y2 + 20), group, font=font(27, True), fill=BLACK)
        for si, s in enumerate(series):
            v = values[si][gi]
            bx1 = center - len(series) * bar_w / 2 + si * bar_w
            by = y2 - (v - ymin) / (ymax - ymin) * (y2 - y1)
            d.rectangle((bx1, by, bx1 + bar_w * .78, y2), fill=colors[si], outline=WHITE)
            d.text((bx1, by - 35), f"{v:.3f}", font=font(23, True), fill=BLACK)
    lx = 260
    for i, s in enumerate(series):
        d.rectangle((lx, 930, lx + 35, 965), fill=colors[i]); d.text((lx + 45, 928), s, font=font(26), fill=BLACK); lx += 300
    save(im, name)


def class_distribution():
    counts = [[26, 38, 15], [20, 84, 26], [26, 42, 28]]
    im, d = new_image(1800, 1050); title(d, "Phân bố lớp trong locked test", "Số lượng mẫu thực tế theo nhãn Low, Medium, High")
    plot = (170, 160, 1690, 850); axes(d, plot, 0, 90, 6)
    colors = ["#C0504D", GOLD, GREEN]; groups = ["Student-Mat", "Student-Por", "xAPI"]
    for gi, g in enumerate(groups):
        cx = 380 + gi * 500
        for si, v in enumerate(counts[gi]):
            x = cx + (si - 1) * 105
            y = 850 - v / 90 * 690
            d.rectangle((x, y, x + 78, 850), fill=colors[si])
            d.text((x + 20, y - 34), str(v), font=font(25, True), fill=BLACK)
        d.text((cx - 100, 875), g, font=font(28, True), fill=BLACK)
    for i, s in enumerate(["Low", "Medium", "High"]):
        d.rectangle((520 + i * 270, 960, 555 + i * 270, 995), fill=colors[i]); d.text((565 + i * 270, 958), s, font=font(26), fill=BLACK)
    save(im, "06_class_distribution.png")


def metric_charts():
    groups = ["Student-Mat", "Student-Por", "xAPI"]
    bar_chart("07_locked_metrics.png", "So sánh metric phân loại trên locked test", groups,
              ["Accuracy", "Precision-Macro", "Recall-Macro", "F1-Macro"],
              [[.8861, .8615, .7604], [.8781, .8110, .7661], [.9170, .8816, .7711], [.8905, .8394, .7663]], .65, .95)
    bar_chart("08_cv_test_gap.png", "So sánh F1-Macro: Cross-validation và Locked test", groups,
              ["Best CV F1", "Locked-test F1"], [[.9112, .8898, .8121], [.8905, .8394, .7663]], .70, .95)
    bar_chart("10_per_class_f1.png", "F1-score theo từng lớp", groups,
              ["Low", "Medium", "High"], [[.89, .74, .82], [.87, .89, .72], [.91, .89, .75]], .60, 1.0)


def confusion_matrices():
    matrices = [
        ("Student-Mat", [[25, 1, 0], [5, 30, 3], [0, 0, 15]]),
        ("Student-Por", [[17, 3, 0], [9, 70, 5], [0, 1, 25]]),
        ("xAPI", [[23, 3, 0], [7, 30, 5], [0, 8, 20]]),
    ]
    im, d = new_image(1900, 1030); title(d, "Ma trận nhầm lẫn trên locked test", "Hàng: nhãn thật; cột: nhãn dự đoán")
    classes = ["Low", "Medium", "High"]
    for pi, (name, mat) in enumerate(matrices):
        ox = 100 + pi * 610; oy = 250; cell = 145
        d.text((ox + 130, 155), name, font=font(34, True), fill=NAVY)
        for j, c in enumerate(classes): d.text((ox + 80 + j * cell, 210), c, font=font(24, True), fill=BLACK)
        maxv = max(max(row) for row in mat)
        for i, row in enumerate(mat):
            d.text((ox - 5, oy + i * cell + 50), classes[i], font=font(24, True), fill=BLACK)
            for j, v in enumerate(row):
                intensity = int(245 - 155 * v / maxv)
                fill = (intensity, intensity + 5, 255)
                xy = (ox + 80 + j * cell, oy + i * cell, ox + 80 + (j + 1) * cell, oy + (i + 1) * cell)
                d.rectangle(xy, fill=fill, outline=WHITE, width=4)
                text_center(d, xy, str(v), 38, True, NAVY)
    save(im, "09_confusion_matrices.png")


def feature_importance():
    im, d = new_image(1900, 1200); title(d, "Permutation feature importance", "Mức giảm F1-Macro khi hoán vị từng đặc trưng; hiển thị 6 đặc trưng cao nhất")
    datasets = ["student-mat", "student-por", "xapi"]
    labels = ["Student-Mat", "Student-Por", "xAPI"]
    for pi, (ds, label) in enumerate(zip(datasets, labels)):
        df = pd.read_csv(ROOT / "reports/final/explanations" / f"{ds}_3class_feature_importance.csv").head(6).iloc[::-1]
        ox = 90 + pi * 610; oy = 220
        d.text((ox + 130, 140), label, font=font(34, True), fill=NAVY)
        maxv = max(float(df.Importance.max()), .001)
        for i, row in enumerate(df.itertuples()):
            y = oy + i * 125
            raw_name = str(row.Feature)
            name = raw_name if len(raw_name) <= 17 else raw_name[:16] + "…"
            d.text((ox, y + 20), name, font=font(21), fill=BLACK)
            length = 330 * max(float(row.Importance), 0) / maxv
            d.rectangle((ox + 210, y + 12, ox + 210 + length, y + 55), fill=BLUE)
            d.text((ox + 220 + length, y + 15), f"{row.Importance:.3f}", font=font(21), fill=GRAY)
    text_center(d, (120, 1030, 1780, 1140), "Giá trị bằng 0 hoặc âm cho thấy đặc trưng không cải thiện F1 trong phép đo hoán vị cụ thể", 27, False, GRAY)
    save(im, "11_feature_importance.png")


def confidence_and_risk():
    datasets = ["student-mat", "student-por", "xapi"]
    labels = ["Student-Mat", "Student-Por", "xAPI"]
    bins = [0.0, .5, .6, .7, .8, .9, 1.001]
    series = []
    for ds in datasets:
        df = pd.read_csv(ROOT / "reports/final/predictions" / f"{ds}_3class_predictions.csv")
        series.append(pd.cut(df.Confidence, bins=bins, right=False).value_counts().sort_index().tolist())
    im, d = new_image(1800, 1050); title(d, "Phân bố độ tin cậy dự đoán", "Tần suất confidence của ensemble theo các khoảng xác suất")
    plot = (170, 160, 1690, 840); maxv = max(max(x) for x in series); axes(d, plot, 0, maxv + 5, 5)
    names = ["<0.50", "0.50-0.59", "0.60-0.69", "0.70-0.79", "0.80-0.89", "≥0.90"]
    colors = [BLUE, GREEN, GOLD]
    for bi, name in enumerate(names):
        cx = 260 + bi * 245
        for di in range(3):
            v = series[di][bi]; x = cx + di * 52; y = 840 - v / (maxv + 5) * 680
            d.rectangle((x, y, x + 42, 840), fill=colors[di]); d.text((x, y - 26), str(v), font=font(18, True), fill=BLACK)
        d.text((cx - 10, 865), name, font=font(21), fill=BLACK)
    for i, label in enumerate(labels):
        d.rectangle((520 + i * 300, 960, 555 + i * 300, 995), fill=colors[i]); d.text((565 + i * 300, 958), label, font=font(25), fill=BLACK)
    save(im, "12_confidence_distribution.png")

    risk_series = []
    risk_names = ["stable", "moderate", "high"]
    for ds in datasets:
        lp = pd.read_csv(ROOT / "reports/final/recommendations" / f"{ds}_3class_learning_paths.csv")
        vc = lp.risk_band.value_counts()
        risk_series.append([int(vc.get(x, 0)) for x in risk_names])
    im, d = new_image(1800, 1050); title(d, "Phân bố mức rủi ro của Learning Path Engine")
    plot = (170, 160, 1690, 840); maxv = max(max(x) for x in risk_series); axes(d, plot, 0, maxv + 10, 5)
    rcolors = [GREEN, GOLD, "#C0504D"]
    for di, label in enumerate(labels):
        cx = 380 + di * 500
        for ri, risk in enumerate(risk_names):
            v = risk_series[di][ri]; x = cx + (ri - 1) * 110; y = 840 - v / (maxv + 10) * 680
            d.rectangle((x, y, x + 80, 840), fill=rcolors[ri]); d.text((x + 18, y - 30), str(v), font=font(23, True), fill=BLACK)
        d.text((cx - 100, 870), label, font=font(28, True), fill=BLACK)
    for i, risk in enumerate(risk_names):
        d.rectangle((550 + i * 250, 960, 585 + i * 250, 995), fill=rcolors[i]); d.text((595 + i * 250, 958), risk, font=font(25), fill=BLACK)
    save(im, "13_risk_band_distribution.png")


def main():
    pipeline(); architecture(); leakage_control(); rule_engine(); database_schema(); class_distribution(); metric_charts(); confusion_matrices(); feature_importance(); confidence_and_risk()
    print(f"Generated {len(list(OUT.glob('*.png')))} figures in {OUT}")


if __name__ == "__main__":
    main()
