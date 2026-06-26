from __future__ import annotations

import csv
import json
import re
from pathlib import Path

import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = Path(__file__).resolve().parent


def parse_final_prediction_rows() -> list[dict[str, str]]:
    path = ROOT / "reports" / "final" / "FINAL_PROJECT_STATUS.md"
    rows: list[dict[str, str]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.startswith("|"):
            continue
        if "Dataset" in line or "---" in line:
            continue
        cells = [cell.strip() for cell in line.strip("|").split("|")]
        if len(cells) != 7:
            continue
        rows.append(
            {
                "dataset": cells[0],
                "scenario": cells[1],
                "model": cells[2],
                "prediction_mode": cells[3],
                "macro_f1": cells[4],
                "recall_low": cells[5],
                "f1_low": cells[6],
            }
        )
    return rows


def load_baseline_rows() -> list[dict[str, str]]:
    path = ROOT / "reports" / "final" / "final_baseline_comparison.csv"
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def load_recommender_rows() -> list[dict[str, float | str]]:
    rows: list[dict[str, float | str]] = []
    for dataset in ["xapi", "student-por"]:
        path = ROOT / "outputs" / "recommender" / dataset / "recommender_metrics.json"
        data = json.loads(path.read_text(encoding="utf-8"))
        rows.append(
            {
                "dataset": dataset,
                "risk_macro_f1": data["risk_diagnosis"]["f1_macro"],
                "risk_micro_f1": data["risk_diagnosis"]["f1_micro"],
                "precision_at_3": data["ranking"]["precision_at_3"],
                "recall_at_3": data["ranking"]["recall_at_3"],
                "ndcg_at_3": data["ranking"]["ndcg_at_3"],
                "coverage_at_3": data["ranking"]["coverage_at_3"],
                "risk_coverage": data["path_quality"]["risk_coverage_rate"],
                "workload_std": data["path_quality"]["workload_balance_std"],
                "difficulty_progression": data["path_quality"]["difficulty_progression_rate"],
                "prereq_violation": data["path_quality"]["prerequisite_violation_rate"],
            }
        )
    return rows


def bar_chart(path: Path, title: str, subtitle: str, categories: list[str], series: dict[str, list[float]], colors: dict[str, str]) -> None:
    fig, ax = plt.subplots(figsize=(10.5, 5.8), dpi=140)
    x = range(len(categories))
    width = 0.78 / max(len(series), 1)
    offsets = [(-0.39 + width / 2) + idx * width for idx in range(len(series))]
    for offset, (name, values) in zip(offsets, series.items()):
        positions = [idx + offset for idx in x]
        ax.bar(positions, values, width=width, label=name, color=colors.get(name, "#2563EB"))
        for px, value in zip(positions, values):
            ax.text(px, value + 0.015, f"{value:.3f}", ha="center", va="bottom", fontsize=8)
    ax.set_title(title, loc="left", fontsize=14, fontweight="bold")
    ax.text(0, 1.03, subtitle, transform=ax.transAxes, fontsize=9, color="#4B5563")
    ax.set_xticks(list(x))
    ax.set_xticklabels(categories, rotation=16, ha="right")
    ax.set_ylim(0, max(1.05, max(max(values) for values in series.values()) + 0.12))
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False, ncol=min(3, len(series)), loc="upper right")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def horizontal_bar(path: Path, title: str, subtitle: str, categories: list[str], values: list[float], color: str) -> None:
    fig, ax = plt.subplots(figsize=(9, 4.8), dpi=140)
    y = range(len(categories))
    ax.barh(list(y), values, color=color)
    for yi, value in zip(y, values):
        ax.text(value + 0.012, yi, f"{value:.3f}", va="center", fontsize=9)
    ax.set_title(title, loc="left", fontsize=14, fontweight="bold")
    ax.text(0, 1.03, subtitle, transform=ax.transAxes, fontsize=9, color="#4B5563")
    ax.set_yticks(list(y))
    ax.set_yticklabels(categories)
    ax.set_xlim(0, max(1.05, max(values) + 0.12))
    ax.invert_yaxis()
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def flow_diagram(path: Path, title: str, subtitle: str, labels: list[str], colors: list[str]) -> None:
    fig, ax = plt.subplots(figsize=(12, 3.8), dpi=140)
    ax.axis("off")
    ax.set_title(title, loc="left", fontsize=14, fontweight="bold")
    ax.text(0, 0.92, subtitle, transform=ax.transAxes, fontsize=9, color="#4B5563")
    count = len(labels)
    box_w = 0.86 / count
    y = 0.45
    for idx, label in enumerate(labels):
        x = 0.04 + idx * (0.92 / count)
        rect = plt.Rectangle((x, y), box_w, 0.24, facecolor=colors[idx % len(colors)], edgecolor="#111827", linewidth=0.8)
        ax.add_patch(rect)
        ax.text(x + box_w / 2, y + 0.12, label, ha="center", va="center", fontsize=8.5, wrap=True)
        if idx < count - 1:
            ax.annotate(
                "",
                xy=(x + box_w + 0.018, y + 0.12),
                xytext=(x + box_w + 0.002, y + 0.12),
                arrowprops={"arrowstyle": "->", "lw": 1.1, "color": "#111827"},
            )
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def write_manifest(rows: list[dict[str, str]]) -> None:
    manifest_path = OUT_DIR / "figure_manifest.csv"
    fieldnames = [
        "id",
        "file",
        "title",
        "chapter",
        "source_data",
        "script",
        "caption_vi",
        "status",
        "authenticity_note",
    ]
    with manifest_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    readme_lines = [
        "# Verified Figure Pack",
        "",
        "Các hình trong thư mục này được tạo từ artifact có sẵn trong repository, không dùng dữ liệu giả hoặc ảnh AI.",
        "",
    ]
    for row in rows:
        readme_lines.append(f"- `{row['file']}`: {row['caption_vi']} Source: `{row['source_data']}`.")
    (OUT_DIR / "README.md").write_text("\n".join(readme_lines) + "\n", encoding="utf-8")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    pred = parse_final_prediction_rows()
    baseline = load_baseline_rows()
    rec = load_recommender_rows()

    manifest: list[dict[str, str]] = []

    categories = [f"{r['dataset']} {r['scenario']}" for r in pred]
    bar_chart(
        OUT_DIR / "fig_01_prediction_metrics.png",
        "Kết quả dự đoán final theo dataset/scenario",
        "Nguồn: reports/final/FINAL_PROJECT_STATUS.md",
        categories,
        {
            "Macro F1": [float(r["macro_f1"]) for r in pred],
            "Recall Low": [float(r["recall_low"]) for r in pred],
            "F1 Low": [float(r["f1_low"]) for r in pred],
        },
        {"Macro F1": "#2563EB", "Recall Low": "#16A34A", "F1 Low": "#F97316"},
    )
    manifest.append({
        "id": "FIG-PRED-01",
        "file": "fig_01_prediction_metrics.png",
        "title": "Kết quả dự đoán final",
        "chapter": "Chương 4",
        "source_data": "reports/final/FINAL_PROJECT_STATUS.md",
        "script": "report_context/figures/create_verified_figures.py",
        "caption_vi": "So sánh Macro F1, Recall Low và F1 Low của các mô hình dự đoán final theo dataset/scenario.",
        "status": "VERIFIED_DATA_BACKED",
        "authenticity_note": "Số liệu lấy trực tiếp từ bảng final status; student rows chỉ có bằng chứng summary final, không có CSV per-run riêng.",
    })

    xapi_rows = [row for row in baseline if row["dataset"].lower() == "xapi"]
    bar_chart(
        OUT_DIR / "fig_02_xapi_baseline_comparison.png",
        "So sánh xAPI deep final với baseline",
        "Nguồn: reports/final/final_baseline_comparison.csv",
        [row["model_type"].capitalize() for row in xapi_rows],
        {"Macro F1": [float(row["macro_f1"]) for row in xapi_rows]},
        {"Macro F1": "#7C3AED"},
    )
    manifest.append({
        "id": "FIG-PRED-02",
        "file": "fig_02_xapi_baseline_comparison.png",
        "title": "xAPI deep vs baseline",
        "chapter": "Chương 4",
        "source_data": "reports/final/final_baseline_comparison.csv",
        "script": "report_context/figures/create_verified_figures.py",
        "caption_vi": "So sánh Macro F1 giữa mô hình deep xAPI final và RandomForestClassifier baseline.",
        "status": "VERIFIED_DATA_BACKED",
        "authenticity_note": "Baseline chỉ là đối chứng, không dùng làm teacher/distillation/pseudo-label.",
    })

    bar_chart(
        OUT_DIR / "fig_03_low_class_focus.png",
        "Phân tích riêng lớp Low",
        "Nguồn: reports/final/FINAL_PROJECT_STATUS.md",
        categories,
        {
            "Recall Low": [float(r["recall_low"]) for r in pred],
            "F1 Low": [float(r["f1_low"]) for r in pred],
        },
        {"Recall Low": "#16A34A", "F1 Low": "#F97316"},
    )
    manifest.append({
        "id": "FIG-PRED-03",
        "file": "fig_03_low_class_focus.png",
        "title": "Metric lớp Low",
        "chapter": "Chương 4",
        "source_data": "reports/final/FINAL_PROJECT_STATUS.md",
        "script": "report_context/figures/create_verified_figures.py",
        "caption_vi": "Recall Low và F1 Low cho thấy năng lực phát hiện nhóm sinh viên có nguy cơ kết quả thấp.",
        "status": "VERIFIED_DATA_BACKED",
        "authenticity_note": "Không diễn giải Recall Low riêng lẻ là bằng chứng mô hình tốt nhất.",
    })

    macro_sorted = sorted(pred, key=lambda row: float(row["macro_f1"]), reverse=True)
    horizontal_bar(
        OUT_DIR / "fig_04_macro_f1_ranking.png",
        "Xếp hạng Macro F1 của final champions",
        "Nguồn: reports/final/FINAL_PROJECT_STATUS.md",
        [f"{r['dataset']} {r['scenario']}" for r in macro_sorted],
        [float(r["macro_f1"]) for r in macro_sorted],
        "#0F766E",
    )
    manifest.append({
        "id": "FIG-PRED-04",
        "file": "fig_04_macro_f1_ranking.png",
        "title": "Xếp hạng Macro F1",
        "chapter": "Chương 4",
        "source_data": "reports/final/FINAL_PROJECT_STATUS.md",
        "script": "report_context/figures/create_verified_figures.py",
        "caption_vi": "Xếp hạng các cấu hình final theo Macro F1.",
        "status": "VERIFIED_DATA_BACKED",
        "authenticity_note": "Dùng để mô tả thứ tự hiệu quả, không khẳng định deep model thắng baseline ở mọi dataset.",
    })

    rec_categories = [str(row["dataset"]) for row in rec]
    bar_chart(
        OUT_DIR / "fig_05_recommender_offline_metrics.png",
        "Đánh giá offline RA-HLPR",
        "Nguồn: outputs/recommender/*/recommender_metrics.json",
        rec_categories,
        {
            "Risk Macro F1": [float(row["risk_macro_f1"]) for row in rec],
            "Risk Micro F1": [float(row["risk_micro_f1"]) for row in rec],
            "Precision@3": [float(row["precision_at_3"]) for row in rec],
            "Recall@3": [float(row["recall_at_3"]) for row in rec],
            "NDCG@3": [float(row["ndcg_at_3"]) for row in rec],
            "Risk Coverage": [float(row["risk_coverage"]) for row in rec],
        },
        {
            "Risk Macro F1": "#2563EB",
            "Risk Micro F1": "#38BDF8",
            "Precision@3": "#16A34A",
            "Recall@3": "#84CC16",
            "NDCG@3": "#F97316",
            "Risk Coverage": "#E11D48",
        },
    )
    manifest.append({
        "id": "FIG-REC-01",
        "file": "fig_05_recommender_offline_metrics.png",
        "title": "RA-HLPR offline metrics",
        "chapter": "Chương 5",
        "source_data": "outputs/recommender/xapi/recommender_metrics.json; outputs/recommender/student-por/recommender_metrics.json",
        "script": "report_context/figures/create_verified_figures.py",
        "caption_vi": "Tổng hợp các chỉ số đánh giá offline của RA-HLPR trên xAPI và student-por.",
        "status": "VERIFIED_DATA_BACKED",
        "authenticity_note": "Reference là weak-supervision/rule-based, không phải phản hồi người dùng thật.",
    })

    bar_chart(
        OUT_DIR / "fig_06_risk_diagnosis_metrics.png",
        "Chẩn đoán rủi ro RA-HLPR",
        "Nguồn: outputs/recommender/*/recommender_metrics.json",
        rec_categories,
        {
            "Risk Macro F1": [float(row["risk_macro_f1"]) for row in rec],
            "Risk Micro F1": [float(row["risk_micro_f1"]) for row in rec],
        },
        {"Risk Macro F1": "#2563EB", "Risk Micro F1": "#38BDF8"},
    )
    manifest.append({
        "id": "FIG-REC-02",
        "file": "fig_06_risk_diagnosis_metrics.png",
        "title": "Risk diagnosis",
        "chapter": "Chương 5",
        "source_data": "outputs/recommender/xapi/recommender_metrics.json; outputs/recommender/student-por/recommender_metrics.json",
        "script": "report_context/figures/create_verified_figures.py",
        "caption_vi": "Risk Macro F1 và Risk Micro F1 của đầu chẩn đoán rủi ro.",
        "status": "VERIFIED_DATA_BACKED",
        "authenticity_note": "Metric đo fidelity với weak labels tạo từ rule quan sát được.",
    })

    bar_chart(
        OUT_DIR / "fig_07_ranking_metrics.png",
        "Xếp hạng intervention",
        "Nguồn: outputs/recommender/*/recommender_metrics.json",
        rec_categories,
        {
            "Precision@3": [float(row["precision_at_3"]) for row in rec],
            "Recall@3": [float(row["recall_at_3"]) for row in rec],
            "NDCG@3": [float(row["ndcg_at_3"]) for row in rec],
            "Coverage@3": [float(row["coverage_at_3"]) for row in rec],
        },
        {"Precision@3": "#16A34A", "Recall@3": "#84CC16", "NDCG@3": "#F97316", "Coverage@3": "#E11D48"},
    )
    manifest.append({
        "id": "FIG-REC-03",
        "file": "fig_07_ranking_metrics.png",
        "title": "Ranking metrics",
        "chapter": "Chương 5",
        "source_data": "outputs/recommender/xapi/recommender_metrics.json; outputs/recommender/student-por/recommender_metrics.json",
        "script": "report_context/figures/create_verified_figures.py",
        "caption_vi": "Precision@3, Recall@3, NDCG@3 và Coverage@3 của bước xếp hạng can thiệp.",
        "status": "VERIFIED_DATA_BACKED",
        "authenticity_note": "Đánh giá offline theo catalog mapping/risk reference.",
    })

    bar_chart(
        OUT_DIR / "fig_08_path_quality_metrics.png",
        "Chất lượng lộ trình 4 tuần",
        "Nguồn: outputs/recommender/*/recommender_metrics.json",
        rec_categories,
        {
            "Risk Coverage": [float(row["risk_coverage"]) for row in rec],
            "Difficulty Progression": [float(row["difficulty_progression"]) for row in rec],
            "Prereq Violation": [float(row["prereq_violation"]) for row in rec],
        },
        {"Risk Coverage": "#E11D48", "Difficulty Progression": "#7C3AED", "Prereq Violation": "#64748B"},
    )
    manifest.append({
        "id": "FIG-REC-04",
        "file": "fig_08_path_quality_metrics.png",
        "title": "Path quality",
        "chapter": "Chương 5",
        "source_data": "outputs/recommender/xapi/recommender_metrics.json; outputs/recommender/student-por/recommender_metrics.json",
        "script": "report_context/figures/create_verified_figures.py",
        "caption_vi": "Các chỉ số bao phủ rủi ro, tiến triển độ khó và vi phạm prerequisite của lộ trình 4 tuần.",
        "status": "VERIFIED_DATA_BACKED",
        "authenticity_note": "Không đo tác động học tập thực tế sau can thiệp.",
    })

    flow_diagram(
        OUT_DIR / "fig_09_pipeline_overview.png",
        "Pipeline tổng thể đã xác minh",
        "Nguồn: src/data_pipeline.py, src/models*.py, scripts/run_recommender_pipeline.py",
        [
            "Raw datasets",
            "Preprocessing",
            "Feature / sequence construction",
            "CNN-BiLSTM / gated fusion",
            "Risk diagnosis",
            "Intervention ranking",
            "4-week path",
            "Evaluation / reports",
        ],
        ["#DBEAFE", "#DCFCE7", "#FEF3C7", "#FCE7F3", "#EDE9FE", "#FFEDD5", "#ECFDF5", "#F8FAFC"],
    )
    manifest.append({
        "id": "FIG-SYS-01",
        "file": "fig_09_pipeline_overview.png",
        "title": "Pipeline overview",
        "chapter": "Chương 3",
        "source_data": "src/data_pipeline.py; src/models/models.py; src/models_v27.py; scripts/run_recommender_pipeline.py",
        "script": "report_context/figures/create_verified_figures.py",
        "caption_vi": "Pipeline hệ thống từ dữ liệu thô đến dự đoán, chẩn đoán rủi ro, xếp hạng can thiệp và báo cáo.",
        "status": "VERIFIED_CODE_BACKED",
        "authenticity_note": "Sơ đồ cấu trúc dựa trên source code; không biểu diễn metric mới.",
    })

    flow_diagram(
        OUT_DIR / "fig_10_ra_hlpr_flow.png",
        "RA-HLPR: từ xác suất dự đoán đến lộ trình",
        "Nguồn: scripts/run_recommender_pipeline.py và src/recommender/",
        [
            "Class probabilities",
            "Weak risk labels",
            "RiskDiagnosisHead",
            "CandidateGenerator",
            "HybridScorer",
            "PathPlanner",
            "4-week learning path",
        ],
        ["#DBEAFE", "#FDE68A", "#DCFCE7", "#FFEDD5", "#FCE7F3", "#EDE9FE", "#ECFDF5"],
    )
    manifest.append({
        "id": "FIG-REC-05",
        "file": "fig_10_ra_hlpr_flow.png",
        "title": "RA-HLPR flow",
        "chapter": "Chương 5",
        "source_data": "scripts/run_recommender_pipeline.py; src/recommender/risk_rules.py; src/recommender/risk_head.py; src/recommender/candidate_generator.py; src/recommender/hybrid_scorer.py; src/recommender/path_planner.py",
        "script": "report_context/figures/create_verified_figures.py",
        "caption_vi": "Luồng RA-HLPR dùng xác suất dự đoán và rủi ro quan sát được để tạo lộ trình học tập 4 tuần.",
        "status": "VERIFIED_CODE_BACKED",
        "authenticity_note": "Sơ đồ không mô tả collaborative filtering; repo không có user-item feedback.",
    })

    write_manifest(manifest)


if __name__ == "__main__":
    main()
