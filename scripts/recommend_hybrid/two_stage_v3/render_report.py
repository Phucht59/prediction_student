"""Render the final Vietnamese Two-Stage V3 scientific report."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "artifacts/recommend_hybrid/two_stage_v3"
REPORT = ROOT / "reports/recommend_hybrid/TWO_STAGE_V3_FINAL_RESULTS_VI.md"


def _read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def _fmt(value: Any) -> str:
    return f"{value:.4f}" if isinstance(value, float) else str(value)


def main() -> None:
    diagnostic = _read(OUT / "DIAGNOSTIC.json")
    results = _read(OUT / "final_oof/NESTED_OOF_RESULTS.json")
    bootstrap = _read(OUT / "final_oof/BOOTSTRAP.json")
    verification = _read(OUT / "final_oof/VERIFICATION.json")
    release = _read(OUT / "TWO_STAGE_V3_RELEASE.json")
    overall = results.get("overall", {})
    folds = results.get("folds", [])

    lines = [
        "# Kết quả module khuyến nghị Two-Stage V3 tích hợp Hybrid",
        "",
        "## 1. Lý do thay đổi kiến trúc",
        "",
        "Hybrid-only deterministic trước đó đạt Precision@1 0,2711. Diagnostic cho thấy lỗi chính nằm ở tầng quyết định có nên phát khuyến nghị hay không, không chỉ ở việc xếp hạng action.",
        "",
        f"- Stage A cũ — precision: {_fmt(diagnostic.get('overall', {}).get('recommendability_precision', 'NOT_AVAILABLE'))}",
        f"- Stage A cũ — recall: {_fmt(diagnostic.get('overall', {}).get('recommendability_recall', 'NOT_AVAILABLE'))}",
        f"- Stage B cũ — conditional Precision@1: {_fmt(diagnostic.get('overall', {}).get('conditional_action_precision_issued_positive', 'NOT_AVAILABLE'))}",
        f"- End-to-end cũ: {_fmt(diagnostic.get('overall', {}).get('end_to_end_precision_at_1', 'NOT_AVAILABLE'))}",
        "",
        "## 2. Kiến trúc V3",
        "",
        "Backbone dự đoán là residual CNN–BiLSTM 160.492 tham số đã đóng băng. Hệ thống tái sử dụng student-state embedding 64 chiều và tabular-expert embedding 32 chiều, sau đó học hai head tích hợp: recommendability và conditional action scoring.",
        "",
        "```text",
        "Frozen residual CNN–BiLSTM",
        "→ 64-D student state + 32-D tabular expert",
        "→ Stage A recommendability head",
        "→ Stage B conditional action head",
        "→ selective abstention",
        "→ safety / prerequisite / workload constraints",
        "```",
        "",
        "Không sử dụng XGBoost, LightGBM, LambdaMART hoặc một ML ranker tách rời.",
        "",
        "## 3. Kết quả held-out OOF",
        "",
        f"- Groups: {_fmt(overall.get('groups', 'NOT_AVAILABLE'))}",
        f"- Learners: {_fmt(overall.get('learners', 'NOT_AVAILABLE'))}",
        f"- Positive groups: {_fmt(overall.get('positive_groups', 'NOT_AVAILABLE'))}",
        f"- Issued groups: {_fmt(overall.get('issued_groups', 'NOT_AVAILABLE'))}",
        f"- Stage A precision: {_fmt(overall.get('stage_a_precision', 'NOT_AVAILABLE'))}",
        f"- Stage A recall / positive-group coverage: {_fmt(overall.get('stage_a_recall', overall.get('positive_group_coverage', 'NOT_AVAILABLE')))}",
        f"- Stage B conditional Precision@1: {_fmt(overall.get('stage_b_conditional_precision_at_1', 'NOT_AVAILABLE'))}",
        f"- Stage B Precision@1 trên toàn bộ positive groups: {_fmt(overall.get('conditional_precision_at_1_all_positive', 'NOT_AVAILABLE'))}",
        f"- NDCG@3: {_fmt(overall.get('ndcg_at_3', 'NOT_AVAILABLE'))}",
        f"- MRR: {_fmt(overall.get('mrr', 'NOT_AVAILABLE'))}",
        f"- End-to-end Precision@1: {_fmt(overall.get('end_to_end_precision_at_1', 'NOT_AVAILABLE'))}",
        f"- Abstention rate: {_fmt(overall.get('abstention_rate', 'NOT_AVAILABLE'))}",
        f"- Action diversity: {_fmt(overall.get('action_diversity', 'NOT_AVAILABLE'))}",
        f"- Top-action concentration: {_fmt(overall.get('top_action_concentration', 'NOT_AVAILABLE'))}",
        "",
        "## 4. Outer-fold stability",
        "",
    ]
    if folds:
        for row in folds:
            lines.append(
                "- Fold {fold}: end-to-end P@1={p}, coverage={c}, conditional P@1={conditional}".format(
                    fold=row.get("outer_fold"),
                    p=_fmt(row.get("end_to_end_precision_at_1", "NOT_AVAILABLE")),
                    c=_fmt(row.get("positive_group_coverage", "NOT_AVAILABLE")),
                    conditional=_fmt(
                        row.get("stage_b_conditional_precision_at_1", "NOT_AVAILABLE")
                    ),
                )
            )
    else:
        lines.append("- NOT_AVAILABLE")

    lines.extend(
        [
            "",
            "## 5. Bootstrap theo sinh viên",
            "",
            f"- End-to-end Precision@1 95% CI: [{_fmt(bootstrap.get('end_to_end_precision_at_1', {}).get('lower_95', 'NOT_AVAILABLE'))}, {_fmt(bootstrap.get('end_to_end_precision_at_1', {}).get('upper_95', 'NOT_AVAILABLE'))}]",
            f"- Coverage 95% CI: [{_fmt(bootstrap.get('positive_group_coverage', {}).get('lower_95', 'NOT_AVAILABLE'))}, {_fmt(bootstrap.get('positive_group_coverage', {}).get('upper_95', 'NOT_AVAILABLE'))}]",
            f"- Conditional Precision@1 95% CI: [{_fmt(bootstrap.get('stage_b_conditional_precision_at_1', {}).get('lower_95', 'NOT_AVAILABLE'))}, {_fmt(bootstrap.get('stage_b_conditional_precision_at_1', {}).get('upper_95', 'NOT_AVAILABLE'))}]",
            "",
            "## 6. Safety và reproducibility",
            "",
            f"- Verification: `{verification.get('status', 'NOT_AVAILABLE')}`",
            f"- Frozen prediction backbone: `{verification.get('gates', {}).get('prediction_backbone_frozen', 'NOT_AVAILABLE')}`",
            f"- External ML ranker absent: `{verification.get('gates', {}).get('external_ml_ranker_absent', 'NOT_AVAILABLE')}`",
            f"- Future/protected features absent: `{verification.get('gates', {}).get('future_and_protected_features_absent', 'NOT_AVAILABLE')}`",
            f"- Exact numeric replay: `{verification.get('gates', {}).get('numeric_replay', 'NOT_AVAILABLE')}`",
            f"- Exact decision replay: `{verification.get('gates', {}).get('decision_replay', 'NOT_AVAILABLE')}`",
            "",
            "## 7. Scientific release",
            "",
            f"- Status: `{release.get('status', 'MAIN_EXECUTION_NOT_COMPLETED')}`",
            f"- Main gates pass: `{release.get('main_gates_pass', False)}`",
            f"- Negative controls pass: `{release.get('negative_controls_pass', False)}`",
            f"- Runtime package ready: `{release.get('runtime_package_ready', False)}`",
            f"- Runtime authorized: `{release.get('runtime_authorized', False)}`",
            f"- Thesis-scope completion: `{release.get('thesis_scope_completion', 'RECOMMENDATION_MODULE_NOT_COMPLETE')}`",
            "",
            "## 8. Cách hiểu ngưỡng 80%",
            "",
            "Chỉ được nói mô hình đạt trên 80% khi chính xác metric held-out được nêu đạt trên 0,80. Conditional Precision@1 không được gọi là độ chính xác end-to-end. Coverage tối thiểu 0,50 vẫn là điều kiện bắt buộc để tránh đạt precision cao bằng cách abstain gần như toàn bộ.",
            "",
            "## 9. Giới hạn phát biểu",
            "",
            "Đây là bằng chứng predictive relevance ngoại tuyến trên OULAD, không phải bằng chứng tác động nhân quả, không bảo đảm tăng điểm, chưa phải xác nhận chuyên gia và chưa chứng minh khả năng production.",
            "",
            f"Claim boundary: `{release.get('claim_boundary', 'OFFLINE_PREDICTIVE_RELEVANCE_NOT_CAUSAL_EFFECT')}`",
            "",
        ]
    )
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text("\n".join(lines), encoding="utf-8")
    print(REPORT)


if __name__ == "__main__":
    main()
