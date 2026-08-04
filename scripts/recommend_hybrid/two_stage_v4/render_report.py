"""Render the Vietnamese scientific report for action-aware V4."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "artifacts/recommend_hybrid/two_stage_v4"
V3_RELEASE = (
    ROOT / "artifacts/recommend_hybrid/two_stage_v3/TWO_STAGE_V3_RELEASE.json"
)
REPORT = ROOT / "reports/recommend_hybrid/TWO_STAGE_V4_FINAL_RESULTS_VI.md"


def _read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def _fmt(value: Any) -> str:
    return f"{value:.4f}" if isinstance(value, float) else str(value)


def main() -> None:
    v3 = _read(V3_RELEASE)
    results = _read(OUT / "final_oof/NESTED_OOF_RESULTS.json")
    bootstrap = _read(OUT / "final_oof/BOOTSTRAP.json")
    verification = _read(OUT / "final_oof/VERIFICATION.json")
    release = _read(OUT / "TWO_STAGE_V4_RELEASE.json")
    overall = results.get("overall", {})
    folds = results.get("folds", [])

    lines = [
        "# Kết quả module khuyến nghị Two-Stage V4 Action-Aware",
        "",
        "## 1. Lý do phát triển V4",
        "",
        "V3 đã xếp hạng action rất tốt nhưng end-to-end Precision@1 vẫn bị giới hạn bởi false issue ở recommendability gate. V3 chỉ huấn luyện candidate binary loss trên positive groups, nên action head không bị phạt khi tạo xác suất action cao cho negative groups.",
        "",
        f"- V3 end-to-end Precision@1: {_fmt(v3.get('overall', {}).get('end_to_end_precision_at_1', 'NOT_AVAILABLE'))}",
        f"- V3 Stage A precision: {_fmt(v3.get('overall', {}).get('stage_a_precision', 'NOT_AVAILABLE'))}",
        f"- V3 conditional Precision@1: {_fmt(v3.get('overall', {}).get('stage_b_conditional_precision_at_1', 'NOT_AVAILABLE'))}",
        "",
        "## 2. Kiến trúc V4",
        "",
        "Backbone residual CNN–BiLSTM 160.492 tham số và embedding cache được giữ nguyên. V4 chỉ thay đổi hai neural head tích hợp:",
        "",
        "```text",
        "Frozen residual CNN–BiLSTM",
        "→ direct recommendability head",
        "→ candidate action head học trên mọi valid candidate",
        "→ masked noisy-OR action recommendability",
        "→ direct/action joint gate",
        "→ stage-specific selective thresholds",
        "→ safety / prerequisite / workload constraints",
        "```",
        "",
        "Không sử dụng XGBoost, LightGBM, LambdaMART hoặc external ML ranker.",
        "",
        "## 3. Kết quả held-out OOF",
        "",
        f"- Groups: {_fmt(overall.get('groups', 'NOT_AVAILABLE'))}",
        f"- Learners: {_fmt(overall.get('learners', 'NOT_AVAILABLE'))}",
        f"- Positive groups: {_fmt(overall.get('positive_groups', 'NOT_AVAILABLE'))}",
        f"- Issued groups: {_fmt(overall.get('issued_groups', 'NOT_AVAILABLE'))}",
        f"- Stage A precision: {_fmt(overall.get('stage_a_precision', 'NOT_AVAILABLE'))}",
        f"- Stage A recall / coverage: {_fmt(overall.get('stage_a_recall', overall.get('positive_group_coverage', 'NOT_AVAILABLE')))}",
        f"- Stage B conditional Precision@1: {_fmt(overall.get('stage_b_conditional_precision_at_1', 'NOT_AVAILABLE'))}",
        f"- Ranking-only Precision@1 trên toàn positive groups: {_fmt(overall.get('conditional_precision_at_1_all_positive', 'NOT_AVAILABLE'))}",
        f"- NDCG@3: {_fmt(overall.get('ndcg_at_3', 'NOT_AVAILABLE'))}",
        f"- MRR: {_fmt(overall.get('mrr', 'NOT_AVAILABLE'))}",
        f"- End-to-end Precision@1: {_fmt(overall.get('end_to_end_precision_at_1', 'NOT_AVAILABLE'))}",
        f"- Abstention rate: {_fmt(overall.get('abstention_rate', 'NOT_AVAILABLE'))}",
        f"- Action diversity: {_fmt(overall.get('action_diversity', 'NOT_AVAILABLE'))}",
        f"- Top-action concentration: {_fmt(overall.get('top_action_concentration', 'NOT_AVAILABLE'))}",
        "",
        "## 4. Stage A discrimination",
        "",
    ]
    discrimination = overall.get("stage_a_discrimination", {})
    for source in ("direct", "action_derived", "joint"):
        row = discrimination.get(source, {})
        lines.append(
            f"- {source}: ROC-AUC={_fmt(row.get('roc_auc', 'NOT_AVAILABLE'))}, AP={_fmt(row.get('average_precision', 'NOT_AVAILABLE'))}, Brier={_fmt(row.get('brier_score', 'NOT_AVAILABLE'))}"
        )
    lines.extend(["", "## 5. Outer-fold stability", ""])
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
    lines.extend(["", "## 6. Per-stage", ""])
    for row in overall.get("per_stage", []):
        lines.append(
            "- {stage}: end-to-end P@1={p}, coverage={coverage}, conditional P@1={conditional}".format(
                stage=row.get("stage"),
                p=_fmt(row.get("end_to_end_precision_at_1", "NOT_AVAILABLE")),
                coverage=_fmt(row.get("positive_group_coverage", "NOT_AVAILABLE")),
                conditional=_fmt(row.get("conditional_precision_at_1", "NOT_AVAILABLE")),
            )
        )
    lines.extend(["", "## 7. Per-action", ""])
    for row in overall.get("per_action", []):
        lines.append(
            "- {action}: issued={issued}, precision={p}, conditional precision={conditional}".format(
                action=row.get("action_family"),
                issued=row.get("issued"),
                p=_fmt(row.get("precision", "NOT_AVAILABLE")),
                conditional=_fmt(row.get("conditional_precision", "NOT_AVAILABLE")),
            )
        )
    lines.extend(
        [
            "",
            "## 8. Bootstrap theo sinh viên",
            "",
            f"- End-to-end Precision@1 95% CI: [{_fmt(bootstrap.get('end_to_end_precision_at_1', {}).get('lower_95', 'NOT_AVAILABLE'))}, {_fmt(bootstrap.get('end_to_end_precision_at_1', {}).get('upper_95', 'NOT_AVAILABLE'))}]",
            f"- Coverage 95% CI: [{_fmt(bootstrap.get('positive_group_coverage', {}).get('lower_95', 'NOT_AVAILABLE'))}, {_fmt(bootstrap.get('positive_group_coverage', {}).get('upper_95', 'NOT_AVAILABLE'))}]",
            f"- Conditional Precision@1 95% CI: [{_fmt(bootstrap.get('stage_b_conditional_precision_at_1', {}).get('lower_95', 'NOT_AVAILABLE'))}, {_fmt(bootstrap.get('stage_b_conditional_precision_at_1', {}).get('upper_95', 'NOT_AVAILABLE'))}]",
            "",
            "## 9. Safety và reproducibility",
            "",
            f"- Verification: `{verification.get('status', 'NOT_AVAILABLE')}`",
            f"- Frozen prediction backbone: `{verification.get('gates', {}).get('prediction_backbone_frozen', 'NOT_AVAILABLE')}`",
            f"- All-group candidate supervision: `{verification.get('gates', {}).get('candidate_binary_all_groups', 'NOT_AVAILABLE')}`",
            f"- External ML ranker absent: `{verification.get('gates', {}).get('external_ml_ranker_absent', 'NOT_AVAILABLE')}`",
            f"- Future/protected features absent: `{verification.get('gates', {}).get('future_and_protected_features_absent', 'NOT_AVAILABLE')}`",
            f"- Exact numeric replay: `{verification.get('gates', {}).get('numeric_replay', 'NOT_AVAILABLE')}`",
            f"- Exact decision replay: `{verification.get('gates', {}).get('decision_replay', 'NOT_AVAILABLE')}`",
            "",
            "## 10. Scientific release",
            "",
            f"- Status: `{release.get('status', 'MAIN_EXECUTION_NOT_COMPLETED')}`",
            f"- Main gates pass: `{release.get('main_gates_pass', False)}`",
            f"- Negative controls pass: `{release.get('negative_controls_pass', False)}`",
            f"- Runtime authorized: `{release.get('runtime_authorized', False)}`",
            f"- Thesis-scope completion: `{release.get('thesis_scope_completion', 'RECOMMENDATION_MODULE_NOT_COMPLETE')}`",
            "",
            "## 11. Giới hạn phát biểu",
            "",
            "Conditional Precision@1 không được gọi là độ chính xác end-to-end. Kết quả là predictive relevance ngoại tuyến trên OULAD, không phải tác động nhân quả, không bảo đảm tăng điểm, chưa phải xác nhận chuyên gia và chưa chứng minh production readiness.",
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
