"""Prepare blinded, pre-cutoff expert-review cases from the full cohort."""

from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from evaluate_counterfactual_recommender import (
    STAGES,
    _build_bundle,
    _course_key,
    _load_assessment_dates,
    _observed_features,
)

ROOT = Path(__file__).resolve().parents[2]
FULL = ROOT / "artifacts/recommend_hybrid/counterfactual/full_cohort"
OUT = ROOT / "artifacts/recommend_hybrid/expert_review"
DOCS = ROOT / "docs/recommend_hybrid"
CASE_COUNT = 160
CLAIM_BOUNDARY = "MODEL_ESTIMATED_RISK_REDUCTION_NOT_CAUSAL_EFFECT"
IDENTITY = ["student_key", "course_key", "stage", "fold"]


def _parse(value: Any) -> list[str]:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return []
    try:
        parsed = ast.literal_eval(str(value))
        return [str(item) for item in parsed] if isinstance(parsed, (list, tuple)) else [str(value)]
    except (SyntaxError, ValueError):
        return [str(value)]


def _stable_key(*values: Any) -> str:
    return hashlib.sha256("|".join(map(str, values)).encode("utf-8")).hexdigest()


def _select(rows: pd.DataFrame) -> pd.DataFrame:
    work = rows.copy()
    work["scored_flag"] = work["top_action_id"].notna().map({True: "SCORED", False: "FALLBACK"})
    work["fallback_reason"] = work["fallback_reasons"].map(_parse).map(lambda values: values[0] if values else "NONE")
    work["risk_band"] = pd.qcut(work["baseline_risk"], 3, labels=["LOW", "MEDIUM", "HIGH"], duplicates="drop").astype(str)
    work["presentation"] = work["course_key"].str.rsplit("-", n=1).str[-1]
    work["_stable"] = [_stable_key("expert-review", *row) for row in work[IDENTITY].itertuples(index=False, name=None)]
    work = work.sort_values("_stable")
    chosen: set[tuple[Any, ...]] = set()
    selected_indices: list[int] = []

    def add(group: pd.DataFrame, limit: int) -> None:
        for index in group.index:
            key = tuple(group.loc[index, IDENTITY].tolist())
            if key in chosen:
                continue
            chosen.add(key)
            selected_indices.append(index)
            if len(selected_indices) >= CASE_COUNT or sum(1 for _ in group.index if _ in selected_indices) >= limit:
                break

    # Guarantee balanced representation of stage/status and fallback reason.
    for stage in sorted(work["stage"].unique()):
        for status in ("SCORED", "FALLBACK"):
            add(work.loc[(work["stage"] == stage) & (work["scored_flag"] == status)], 10)
    for stage in sorted(work["stage"].unique()):
        for reason in sorted(work["fallback_reason"].unique()):
            add(work.loc[(work["stage"] == stage) & (work["fallback_reason"] == reason)], 5)
    for band in ("LOW", "MEDIUM", "HIGH"):
        add(work.loc[work["risk_band"] == band], 20)
    for index in work.index:
        if len(selected_indices) >= CASE_COUNT:
            break
        key = tuple(work.loc[index, IDENTITY].tolist())
        if key not in chosen:
            chosen.add(key)
            selected_indices.append(index)
    return work.loc[selected_indices].sort_values("_stable").head(CASE_COUNT).copy()


def _observed_lookup() -> dict[tuple[str, str, str, int], dict[str, Any]]:
    bundle = _build_bundle()
    assessment_dates = _load_assessment_dates()
    lookup: dict[tuple[str, str, str, int], dict[str, Any]] = {}
    for bundle_stage, (canonical_stage, _, _) in STAGES.items():
        data = bundle.stages[bundle_stage]
        frame = data.frame.copy()
        frame["course_key"] = _course_key(frame)
        for index, row in frame.iterrows():
            key = (str(row["base_record_id"]), str(row["course_key"]), str(canonical_stage.value), int(row["outer_fold"]))
            lookup[key] = _observed_features(
                data.sequence[index],
                int(data.lengths[index]),
                cutoff_day=int(row["cutoff_day"]),
                assessment_dates=assessment_dates.get(
                    (str(row["code_module"]), str(row["code_presentation"])),
                    np.array([], dtype=float),
                ),
            )
    return lookup


def _json_safe(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, float) and np.isnan(value):
        return None
    return value


def main() -> int:
    rows = pd.read_parquet(FULL / "evaluation_rows.parquet")
    actions = pd.read_parquet(FULL / "action_scores.parquet")
    selected = _select(rows)
    observed = _observed_lookup()
    cases: list[dict[str, Any]] = []
    for _, row in selected.iterrows():
        identity = tuple(row[key] for key in IDENTITY)
        candidates = actions.loc[
            (actions[IDENTITY] == pd.Series(identity, index=IDENTITY)).all(axis=1)
        ].copy()
        candidates = candidates.sort_values(["utility_status", "risk_reduction", "action_id"], ascending=[True, False, True])
        candidate_payload = []
        for _, action in candidates.iterrows():
            candidate_payload.append({
                "action_id": str(action["action_id"]),
                "status": str(action["utility_status"]),
                "estimated_risk_reduction": _json_safe(action["risk_reduction"]),
                "counterfactual_risk": _json_safe(action["counterfactual_risk"]),
                "evidence_strength": _json_safe(action["evidence_strength"]),
                "uncertainty_penalty": _json_safe(action["uncertainty_penalty"]),
                "workload_minutes": int(action["workload_minutes"]),
                "selected_in_plan": bool(action["selected_in_plan"]),
                "reason_codes": _parse(action["reason_codes"]),
            })
        course_hash = _stable_key("course-presentation", row["course_key"])[:16]
        case = {
            "case_id": _stable_key("case", *identity)[:16],
            "stage": str(row["stage"]),
            "fold": int(row["fold"]),
            "course_presentation_hash": course_hash,
            "baseline_risk": _json_safe(row["baseline_risk"]),
            "risk_band": str(row["risk_band"]),
            "review_status": str(row["scored_flag"]),
            "fallback_reasons": _parse(row["fallback_reasons"]),
            "threshold_crossed": bool(row["threshold_crossed"]),
            "observed_pre_cutoff_signals": observed.get((str(row["student_key"]), str(row["course_key"]), str(row["stage"]), int(row["fold"])), {}),
            "candidate_actions": candidate_payload,
            "selected_action_id": None if pd.isna(row["top_action_id"]) else str(row["top_action_id"]),
            "claim_boundary": CLAIM_BOUNDARY,
            "future_outcomes_included": False,
            "protected_attributes_included": False,
        }
        cases.append(case)
    OUT.mkdir(parents=True, exist_ok=True)
    DOCS.mkdir(parents=True, exist_ok=True)
    (OUT / "EXPERT_REVIEW_CASES.json").write_text(json.dumps(cases, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    csv_rows = []
    for case in cases:
        flat = {key: value for key, value in case.items() if key not in {"observed_pre_cutoff_signals", "candidate_actions"}}
        flat["fallback_reasons"] = json.dumps(flat["fallback_reasons"], ensure_ascii=False)
        flat["observed_pre_cutoff_signals"] = json.dumps(case["observed_pre_cutoff_signals"], ensure_ascii=False)
        flat["candidate_actions"] = json.dumps(case["candidate_actions"], ensure_ascii=False)
        csv_rows.append(flat)
    pd.DataFrame(csv_rows).to_csv(OUT / "EXPERT_REVIEW_CASES.csv", index=False)
    rubric = pd.DataFrame([
        {"criterion_id": "C01", "criterion": "Tín hiệu quan sát trước cutoff có nhất quán với đề xuất không?", "scale": "1=không nhất quán; 5=rất nhất quán"},
        {"criterion_id": "C02", "criterion": "Mức độ phù hợp của hành động với bối cảnh học tập", "scale": "1=không phù hợp; 5=rất phù hợp"},
        {"criterion_id": "C03", "criterion": "Tính khả thi về thời lượng/công sức", "scale": "1=không khả thi; 5=rất khả thi"},
        {"criterion_id": "C04", "criterion": "Tính rõ ràng của lý do/candidate evidence", "scale": "1=không rõ; 5=rất rõ"},
        {"criterion_id": "C05", "criterion": "Mức độ cần xem xét của con người trước khi áp dụng", "scale": "1=không cần; 5=phải xem xét"},
        {"criterion_id": "C06", "criterion": "Rủi ro gây quá tải hoặc khuyến nghị không phù hợp", "scale": "1=rủi ro cao; 5=rủi ro thấp"},
        {"criterion_id": "C07", "criterion": "Nếu fallback/abstain, quyết định không hành động có hợp lý không?", "scale": "1=không hợp lý; 5=rất hợp lý"},
        {"criterion_id": "C08", "criterion": "Tính đa dạng và ưu tiên của candidate set", "scale": "1=không phù hợp; 5=rất phù hợp"},
        {"criterion_id": "C09", "criterion": "Mức độ tin cậy để dùng như hỗ trợ quyết định (không phải nguyên nhân nhân quả)", "scale": "1=không tin cậy; 5=có thể hỗ trợ"},
        {"criterion_id": "C10", "criterion": "Đánh giá tổng thể", "scale": "1=không chấp nhận; 5=chấp nhận"},
    ])
    rubric.to_csv(OUT / "EXPERT_REVIEW_RUBRIC.csv", index=False)
    template = pd.DataFrame({"case_id": [case["case_id"] for case in cases], "reviewer_id": "", "criterion_scores_json": "", "decision": "", "modified_action_id": "", "comments": "", "reviewed_at": ""})
    template.to_csv(OUT / "EXPERT_REVIEW_RESULTS_TEMPLATE.csv", index=False)
    guide = """# Hướng dẫn expert review counterfactual recommender

## Phạm vi

Đánh giá 160 case được lấy mẫu xác định từ full cohort. Case chỉ chứa tín hiệu quan sát trước cutoff, candidate actions và ước lượng rủi ro của model. Không có outcome tương lai, protected attributes hoặc mã định danh người học trong gói review.

## Cách review

1. Đọc `EXPERT_REVIEW_CASES.csv` hoặc JSON và đối chiếu `case_id` với các candidate actions.
2. Chấm 10 tiêu chí trong `EXPERT_REVIEW_RUBRIC.csv` theo thang 1–5.
3. Chọn `ACCEPT`, `ACCEPT_WITH_MODIFICATION`, `REJECT` hoặc `ESCALATE_TO_HUMAN`.
4. Ghi hành động sửa đổi và lý do trong `EXPERT_REVIEW_RESULTS_TEMPLATE.csv`.

Không diễn giải `estimated_risk_reduction` là hiệu quả nhân quả hay outcome thực tế. Fallback/abstain là một kết quả hợp lệ cần được đánh giá riêng.
"""
    (DOCS / "EXPERT_REVIEW_GUIDE.md").write_text(guide, encoding="utf-8")
    protocol = """# Protocol expert review

- Cỡ mẫu: 160 case, lấy mẫu xác định và có phân tầng theo stage, scored/fallback, fallback reason và baseline-risk band.
- Người review không được truy cập outcome tương lai, protected attributes hoặc thông tin ngoài case package trong lúc chấm.
- Mỗi case được ít nhất hai expert review độc lập; bất đồng được chuyển cho adjudicator.
- Báo cáo kết quả theo tỷ lệ ACCEPT/ACCEPT_WITH_MODIFICATION/REJECT/ESCALATE_TO_HUMAN, điểm trung vị từng tiêu chí và độ nhất trí giữa reviewer.
- Không dùng expert review để tuyên bố causal effect; review chỉ đánh giá tính hợp lý, an toàn vận hành và khả năng sử dụng làm decision support.
- Gói này chưa chứa kết quả review; trạng thái hiện tại là `PREPARED_NOT_COMPLETED`.
"""
    (DOCS / "EXPERT_REVIEW_PROTOCOL.md").write_text(protocol, encoding="utf-8")
    print(json.dumps({"status": "PASS", "case_count": len(cases), "scored": int((selected["scored_flag"] == "SCORED").sum()), "fallback": int((selected["scored_flag"] == "FALLBACK").sum())}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
