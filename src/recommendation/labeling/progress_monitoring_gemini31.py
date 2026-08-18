"""Gemini 3.1 Flash-Lite batch contract for A4 Progress Monitoring."""

from __future__ import annotations

import json
from collections.abc import Mapping

from .constants import A4_PROGRESS_GEMINI31_PROMPT_VERSION
from .parser import LabelParseError
from .progress_monitoring import PROGRESS_STATE_FIELDS

PROGRESS_GEMINI31_MODEL = "gemini-3.1-flash-lite"
PROGRESS_GEMINI31_SCHEMA_VERSION = "recommendation.progress_monitoring_gemini31.v1"


def build_progress_monitoring_gemini31_prompt(cases) -> str:
    """Build the same A4 semantics as the existing Progress Monitoring prompt.

    Gemini 3.1 uses a 10-case JSON batch, while the evidence fields and rubric
    remain identical to the existing Progress Monitoring labeling contract.
    """
    case_list = []
    for case in cases:
        case_list.append({"case_id": str(case["case_id"]), **{field: case[field] for field in PROGRESS_STATE_FIELDS}})
    return f"""You are producing weak relevance labels for A4 Progress Monitoring.
Prompt version: {A4_PROGRESS_GEMINI31_PROMPT_VERSION}.

Definition: Progress Monitoring encourages the learner to explicitly review current learning progress,
identify whether they are falling behind, and monitor improvement over time.

Progress Monitoring is NOT Study Planning. A3 Study Planning organizes a future study schedule or rhythm;
A4 inspects current progress and determines whether progress is on track.

Use only the supplied evidence. Do not invent information, infer future information, use outcomes, or
assume unavailable content-level evidence. Evaluate each case independently. Do not return recommendation
prose or explanations.

Allowed evidence fields are exactly:
stage, risk_probability, risk_band, recent_activity, activity_trend, active_days_ratio,
assessment_completion, missing_assessments, course_progress.

Rubric:
0 = NOT_RELEVANT
1 = SLIGHTLY_RELEVANT
2 = RELEVANT
3 = HIGHLY_RELEVANT
ABSTAIN = insufficient observable evidence

Return strict JSON only in this exact shape, with one result per input case and no extra fields:
{{"results":[{{"case_id":"...","labels":{{"A4":{{"label":"0|1|2|3|ABSTAIN"}}}}}}]}}

Input cases:
{json.dumps(case_list, ensure_ascii=False, sort_keys=True, separators=(",", ":"))}
"""


def _normalize_label(value):
    if type(value) is int and value in (0, 1, 2, 3):
        return value
    if isinstance(value, str) and value in {"0", "1", "2", "3"}:
        return int(value)
    if value == "ABSTAIN":
        return value
    raise LabelParseError(f"invalid Progress Monitoring Gemini 3.1 label: {value!r}")


def parse_progress_monitoring_gemini31_response(raw_text: str, expected_case_ids, _feasibility=None) -> dict[str, dict]:
    """Parse only strict A4 JSON results; reject missing, extra, or duplicate cases."""
    try:
        data = json.loads(raw_text)
    except (TypeError, json.JSONDecodeError) as exc:
        raise LabelParseError("Progress Monitoring Gemini 3.1 response is not valid JSON") from exc
    if not isinstance(data, Mapping) or set(data) != {"results"} or not isinstance(data["results"], list):
        raise LabelParseError("Progress Monitoring Gemini 3.1 response must contain only results")
    expected = [str(case_id) for case_id in expected_case_ids]
    if len(expected) != len(set(expected)):
        raise LabelParseError("expected Progress Monitoring case IDs contain duplicates")
    parsed: dict[str, dict] = {}
    for result in data["results"]:
        if not isinstance(result, Mapping) or set(result) != {"case_id", "labels"}:
            raise LabelParseError("each Progress Monitoring result must contain exactly case_id and labels")
        case_id = str(result["case_id"])
        if case_id not in expected:
            raise LabelParseError(f"unexpected Progress Monitoring case_id: {case_id}")
        if case_id in parsed:
            raise LabelParseError(f"duplicate Progress Monitoring case_id: {case_id}")
        labels = result["labels"]
        if not isinstance(labels, Mapping) or set(labels) != {"A4"}:
            raise LabelParseError(f"Progress Monitoring labels must contain exactly A4 for {case_id}")
        item = labels["A4"]
        if not isinstance(item, Mapping) or set(item) != {"label"}:
            raise LabelParseError(f"Progress Monitoring A4 label must contain only label for {case_id}")
        parsed[case_id] = {
            "case_id": case_id,
            "prompt_version": A4_PROGRESS_GEMINI31_PROMPT_VERSION,
            "labels": {"A4": {"label": _normalize_label(item["label"])}},
        }
    if set(parsed) != set(expected):
        raise LabelParseError(f"missing Progress Monitoring case IDs: {sorted(set(expected) - set(parsed))}")
    return parsed
