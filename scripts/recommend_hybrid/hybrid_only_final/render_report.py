"""Render the final Vietnamese hybrid-only recommendation report."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "artifacts/recommend_hybrid/hybrid_only_final"
REPORT = ROOT / "reports/recommend_hybrid/HYBRID_ONLY_FINAL_RESULTS_VI.md"


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def fmt(value) -> str:
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def main() -> None:
    results = read_json(OUT / "evaluation/OOF_RESULTS.json")
    bootstrap = read_json(OUT / "evaluation/BOOTSTRAP.json")
    release = read_json(OUT / "HYBRID_ONLY_RELEASE.json")
    flow = read_json(OUT / "dataset/cohort_flow.json")
    verification = read_json(OUT / "evaluation/VERIFICATION.json")
    folds_path = OUT / "evaluation/FOLD_METRICS.csv"
    folds = pd.read_csv(folds_path) if folds_path.exists() else pd.DataFrame()
    baselines_path = OUT / "evaluation/BASELINE_METRICS.csv"
    baselines = pd.read_csv(baselines_path) if baselines_path.exists() else pd.DataFrame()
    overall = results.get("overall", {})

    lines = [
        "# Kết quả cuối module khuyến nghị Hybrid-only",
        "",
        "## 1. Kiến trúc",
        "",
        "Mô hình học duy nhất là residual CNN–BiLSTM đã đóng băng. Phần khuyến nghị sử dụng policy theo giai đoạn, mô phỏng phản thực bằng chính mô hình hybrid, công thức utility cố định và selective abstention. Không sử dụng XGBoost, LightGBM, LambdaMART hoặc một mô hình xếp hạng học máy thứ hai.",
        "",
        "## 2. Cách hiểu ngưỡng 80%",
        "",
        "Ngưỡng 80% là Precision@1 trên các khuyến nghị thực sự được phát hành, so với silver label hành vi tương lai trực tiếp trong OULAD. Đây không phải tỷ lệ bảo đảm sinh viên tăng điểm và không phải bằng chứng nhân quả.",
        "",
        "## 3. Cohort đánh giá",
        "",
        f"- Transition groups: {fmt(flow.get('transition_groups', 'NOT_AVAILABLE'))}",
        f"- Rankable groups: {fmt(flow.get('rankable_groups', 'NOT_AVAILABLE'))}",
        f"- Candidate rows: {fmt(flow.get('candidate_rows', 'NOT_AVAILABLE'))}",
        f"- Groups có ít nhất một action tích cực tương lai: {fmt(flow.get('groups_with_positive_action', 'NOT_AVAILABLE'))}",
        "",
        "## 4. Kết quả OOF",
        "",
        f"- Precision@1: {fmt(overall.get('precision_at_1', 'NOT_AVAILABLE'))}",
        f"- Actionable coverage: {fmt(overall.get('actionable_coverage', 'NOT_AVAILABLE'))}",
        f"- Issued groups: {fmt(overall.get('issued_groups', 'NOT_AVAILABLE'))}",
        f"- Action diversity: {fmt(overall.get('action_diversity', 'NOT_AVAILABLE'))}",
        f"- Top-action concentration: {fmt(overall.get('top_action_concentration', 'NOT_AVAILABLE'))}",
        "",
        "## 5. Bootstrap theo sinh viên",
        "",
        f"- Precision@1 95% CI: [{fmt(bootstrap.get('precision_at_1', {}).get('lower_95', 'NOT_AVAILABLE'))}, {fmt(bootstrap.get('precision_at_1', {}).get('upper_95', 'NOT_AVAILABLE'))}]",
        f"- Coverage 95% CI: [{fmt(bootstrap.get('actionable_coverage', {}).get('lower_95', 'NOT_AVAILABLE'))}, {fmt(bootstrap.get('actionable_coverage', {}).get('upper_95', 'NOT_AVAILABLE'))}]",
        "",
        "## 6. Outer-fold stability",
        "",
    ]
    if len(folds):
        for row in folds.itertuples():
            lines.append(
                f"- Fold {int(row.outer_fold)}: Precision@1={row.precision_at_1:.4f}, coverage={row.actionable_coverage:.4f}"
            )
    else:
        lines.append("- NOT_AVAILABLE")
    lines.extend(["", "## 7. Deterministic baselines", ""])
    if len(baselines):
        summary = baselines.groupby("method", observed=True)[
            ["precision_at_1", "actionable_coverage"]
        ].mean()
        for method, row in summary.iterrows():
            lines.append(
                f"- {method}: Precision@1={row.precision_at_1:.4f}, coverage={row.actionable_coverage:.4f}"
            )
    else:
        lines.append("- NOT_AVAILABLE")
    lines.extend(
        [
            "",
            "## 8. Safety và reproducibility",
            "",
            f"- Verification status: {verification.get('status', 'NOT_AVAILABLE')}",
            f"- Temporal leakage: {verification.get('gates', {}).get('future_features_in_scoring', 'NOT_AVAILABLE')}",
            f"- Protected-feature exclusion: {verification.get('gates', {}).get('protected_features_in_scoring', 'NOT_AVAILABLE')}",
            f"- Deterministic replay: {verification.get('gates', {}).get('deterministic_replay', 'NOT_AVAILABLE')}",
            "",
            "## 9. Scientific release",
            "",
            f"- Status: `{release.get('status', 'FULL_EXECUTION_NOT_COMPLETED')}`",
            f"- Thesis-scope completion: `{release.get('thesis_scope_completion', 'RECOMMENDATION_MODULE_NOT_COMPLETE')}`",
            f"- Runtime authorized: `{release.get('runtime_authorized', False)}`",
            f"- Claim boundary: `{release.get('claim_boundary', 'HYBRID_MODEL_GUIDED_DECISION_SUPPORT_NOT_CAUSAL_EFFECT')}`",
            "",
            "## 10. Giới hạn phát biểu",
            "",
            "Kết quả chỉ cho phép nói hệ thống chuyển dự đoán rủi ro của hybrid thành khuyến nghị minh bạch và đạt mức phù hợp nhất định với hành vi tương lai quan sát được. Không được nói hệ thống chứng minh tác động nhân quả, bảo đảm tăng điểm, đã được chuyên gia xác nhận hoặc sẵn sàng production.",
            "",
        ]
    )
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text("\n".join(lines), encoding="utf-8")
    print(REPORT)


if __name__ == "__main__":
    main()
