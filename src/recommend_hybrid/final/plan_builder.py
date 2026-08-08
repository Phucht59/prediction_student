"""Structured Learning Plan Builder with scientific claim boundaries."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from .contracts import ActionScore, CanonicalAction, Stage


@dataclass(frozen=True)
class StructuredLearningPlan:
    action: str
    reason: str
    observed_evidence: tuple[str, ...]
    what_to_do: str
    suggested_duration_days: int
    suggested_frequency: str
    measurable_target: str
    reevaluation_time_days: int
    safety_note: str
    claim_boundary: str = (
        "This is an internal model plausibility check, not evidence of causal intervention effectiveness."
    )
    runtime_authorized: bool = False


ACTION_PLANS = {
    CanonicalAction.ASSESSMENT_COMPLETION: {
        "reason": "Có bài đánh giá sắp đến hạn hoặc chưa hoàn thành.",
        "what_to_do": "Xem lại yêu cầu bài nộp, chuẩn bị đề cương và hoàn thiện phần trả lời bài đánh giá.",
        "duration": 7,
        "frequency": "2-3 buổi trong tuần",
        "target": "Nộp bài đánh giá trước hạn 24 giờ.",
        "reeval": 7,
        "safety_note": "Liên hệ giảng viên nếu gặp sự cố kỹ thuật về bài nộp.",
    },
    CanonicalAction.RECOVER_ENGAGEMENT: {
        "reason": "Phát hiện chuỗi ngày gián đoạn tương tác trên hệ thống VLE.",
        "what_to_do": "Đăng nhập lại hệ thống VLE, xem các thông báo mới nhất và truy cập tài liệu môn học.",
        "duration": 5,
        "frequency": "Hàng ngày 15-20 phút",
        "target": "Duy trì ít nhất 4 ngày có hoạt động học tập trên VLE mỗi tuần.",
        "reeval": 7,
        "safety_note": "Nếu có lý do cá nhân gây gián đoạn kéo dài, hãy thông báo cho cố vấn học tập.",
    },
    CanonicalAction.STUDY_REGULARITY: {
        "reason": "Khoảng cách giữa các phiên học lớn và nhịp độ chưa đồng đều.",
        "what_to_do": "Phân chia lịch học thành các phiên nhỏ cố định trong tuần thay vì học dồn.",
        "duration": 14,
        "frequency": "3 buổi mỗi tuần (25-30 phút/buổi)",
        "target": "Giảm khoảng cách tối đa giữa hai phiên học xuống dưới 3 ngày.",
        "reeval": 14,
        "safety_note": "Điều chỉnh thời gian phù hợp với sức khỏe và thời gian biểu cá nhân.",
    },
    CanonicalAction.TARGETED_CONTENT_REVIEW: {
        "reason": "Tỷ lệ bao phủ tài liệu giảng dạy ở các chủ đề trọng tâm còn thấp.",
        "what_to_do": "Đọc lại các bài giảng, bài đọc chính và tóm tắt lại các khái niệm chưa nắm chắc.",
        "duration": 7,
        "frequency": "2 buổi mỗi tuần (45 phút/buổi)",
        "target": "Hoàn thành xem ít nhất 80% tài liệu trọng tâm của chương hiện tại.",
        "reeval": 7,
        "safety_note": "Ưu tiên chất lượng ghi nhớ thay vì chỉ lướt qua tài liệu.",
    },
    CanonicalAction.QUIZ_RETRIEVAL_PRACTICE: {
        "reason": "Cần củng cố kiến thức thông qua luyện tập chủ động.",
        "what_to_do": "Làm các bài trắc nghiệm tự luyện hoặc câu hỏi ôn tập để kiểm tra độ hiểu bài.",
        "duration": 7,
        "frequency": "2-3 lần mỗi tuần",
        "target": "Đạt kết quả trắc nghiệm tự luyện >= 70%.",
        "reeval": 7,
        "safety_note": "Xem lại giải thích đáp án cho các câu trả lời sai.",
    },
}


def build_structured_plan(
    ranked: Sequence[ActionScore],
    stage: Stage,
    observed_evidence_summary: Sequence[str] = (),
) -> StructuredLearningPlan:
    if not ranked:
        raise ValueError("a plan requires at least one ranked action")

    top_action = ranked[0].action
    template = ACTION_PLANS.get(
        top_action,
        ACTION_PLANS[CanonicalAction.TARGETED_CONTENT_REVIEW],
    )

    return StructuredLearningPlan(
        action=top_action.value,
        reason=template["reason"],
        observed_evidence=tuple(observed_evidence_summary),
        what_to_do=template["what_to_do"],
        suggested_duration_days=template["duration"],
        suggested_frequency=template["frequency"],
        measurable_target=template["target"],
        reevaluation_time_days=template["reeval"],
        safety_note=template["safety_note"],
        runtime_authorized=False,
    )
