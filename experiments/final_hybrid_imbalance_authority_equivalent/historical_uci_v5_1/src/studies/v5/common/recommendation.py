from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from typing import Any

import numpy as np


ACTION_CATALOG = {
    "review_prior_material": ("Ôn lại nội dung nền tảng và ghi câu hỏi cụ thể.", 60),
    "contact_advisor": ("Trao đổi với cố vấn để xác nhận khó khăn và điều chỉnh kế hoạch.", 30),
    "practice_assessment": ("Hoàn thành một bài luyện tập ngắn và tự chấm theo rubric.", 75),
    "restore_engagement": ("Thực hiện phiên học ngắn trên hệ thống và ghi lại phần đã hoàn thành.", 45),
    "check_progress": ("Đánh giá tiến độ tuần, trở ngại và bước tiếp theo với cố vấn.", 30),
}


def _hash(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def build_recommendation(
    *,
    case_reference: str,
    dataset: str,
    model_version: str,
    prediction_set: str,
    feature_snapshot: str,
    probabilities: list[float],
    features: dict[str, float | int | None],
    policy_version: str = "v5-rule-policy-1",
    created_at: str | None = None,
) -> dict[str, Any]:
    values = np.asarray(probabilities, dtype=float)
    missing = [name for name, value in features.items() if value is None]
    invalid_probability = (
        values.ndim != 1
        or len(values) not in {2, 3}
        or not np.isfinite(values).all()
        or (values < 0).any()
        or (values > 1).any()
        or not np.isclose(values.sum(), 1.0, atol=1e-5)
    )
    if invalid_probability:
        raise ValueError("Recommendation probability contract failed")
    uncertainty = 1.0 - float(values.max())
    abstained = bool(missing)
    escalation = abstained or uncertainty > 0.40
    risk_probability = float(values[1] if len(values) == 2 else values[0])
    band = "high" if risk_probability >= 0.65 else "medium" if risk_probability >= 0.40 else "low"
    action_codes: list[str] = []
    if not abstained:
        if band in {"high", "medium"}:
            action_codes.extend(["review_prior_material", "contact_advisor"])
        if float(features.get("activity_level", 1) or 0) < 0.35:
            action_codes.append("restore_engagement")
        if dataset in {"student-mat", "student-por"} and float(features.get("grade_trend", 0) or 0) <= 0:
            action_codes.append("practice_assessment")
        action_codes.append("check_progress")
    action_codes = list(dict.fromkeys(action_codes))
    weeks: list[dict[str, Any]] = []
    for week in range(1, 5):
        selected = action_codes[week - 1 :: 4] if action_codes else []
        actions = [
            {"action_code": code, "action": ACTION_CATALOG[code][0], "workload_minutes": ACTION_CATALOG[code][1]}
            for code in selected
        ]
        workload = sum(action["workload_minutes"] for action in actions)
        if workload > 180:
            raise RuntimeError("Recommendation workload exceeded")
        weeks.append({"week": week, "actions": actions, "workload_minutes": workload})
    recommendation = {
        "case_reference": case_reference,
        "dataset": dataset,
        "model_version": model_version,
        "prediction_set": prediction_set,
        "feature_snapshot": feature_snapshot,
        "policy_version": policy_version,
        "created_at": created_at or datetime.now(timezone.utc).isoformat(),
        "risk_probability": risk_probability,
        "risk_band": band,
        "uncertainty": uncertainty,
        "abstained": abstained,
        "abstention_reason": f"missing features: {missing}" if missing else None,
        "escalation_required": escalation,
        "goal": "Khôi phục tiến độ học tập có thể theo dõi và được cố vấn xác nhận.",
        "weeks": weeks,
        "advisor_review": {"required": True, "status": "pending", "decision": None},
        "effectiveness": "NOT_ESTABLISHED",
        "causal_claim": "PROHIBITED",
        "revision_no": 1,
        "supersedes_hash": None,
    }
    recommendation["revision_hash"] = _hash(recommendation)
    return recommendation


def revise_recommendation(previous: dict[str, Any], weeks: list[dict[str, Any]], reason: str) -> dict[str, Any]:
    if _hash({key: value for key, value in previous.items() if key != "revision_hash"}) != previous["revision_hash"]:
        raise ValueError("Previous recommendation revision was mutated")
    revised = dict(previous)
    revised["weeks"] = weeks
    revised["revision_no"] = int(previous["revision_no"]) + 1
    revised["supersedes_hash"] = previous["revision_hash"]
    revised["revision_reason"] = reason
    revised["advisor_review"] = {"required": True, "status": "pending", "decision": None}
    revised.pop("revision_hash", None)
    revised["revision_hash"] = _hash(revised)
    return revised


__all__ = ["ACTION_CATALOG", "build_recommendation", "revise_recommendation"]
