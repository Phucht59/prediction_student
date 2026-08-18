"""Gemma single-case contract for the final A4 Progress Monitoring label."""

from __future__ import annotations

import json
from collections.abc import Mapping

from .constants import A4_PROGRESS_GEMMA_PROMPT_VERSION
from .parser import LabelParseError

PROGRESS_STATE_FIELDS = (
    "stage", "risk_probability", "risk_band", "recent_activity", "activity_trend",
    "active_days_ratio", "assessment_completion", "missing_assessments", "course_progress",
)
PROGRESS_FUNCTION_NAME = "submit_progress_monitoring_label"


def build_progress_monitoring_prompt(case_payload: Mapping) -> str:
    state = {field: case_payload[field] for field in PROGRESS_STATE_FIELDS}
    return f"""You are producing one weak relevance label for A4 Progress Monitoring.
Prompt version: {A4_PROGRESS_GEMMA_PROMPT_VERSION}.

Definition: Progress Monitoring encourages the learner to explicitly review current learning progress,
identify whether they are falling behind, and monitor improvement over time.

Progress Monitoring is NOT Study Planning. A3 Study Planning organizes a future study schedule or rhythm;
A4 inspects current progress and determines whether progress is on track.

Use only the supplied evidence. Do not invent information, infer future information, use outcomes, or
assume unavailable content-level evidence. Return exactly one function call with case_ref `C01` and
one A4 label. Do not return prose.

Rubric: 0=NOT_RELEVANT, 1=SLIGHTLY_RELEVANT, 2=RELEVANT, 3=HIGHLY_RELEVANT,
ABSTAIN=insufficient observable evidence.

Student State (case reference is an alias):
{json.dumps(state, ensure_ascii=False, sort_keys=True)}
"""


def progress_function_declaration() -> dict:
    return {
        "name": PROGRESS_FUNCTION_NAME,
        "description": "Submit exactly one Progress Monitoring relevance label.",
        "parameters": {
            "type": "OBJECT",
            "properties": {
                "cases": {
                    "type": "ARRAY",
                    "items": {
                        "type": "OBJECT",
                        "properties": {
                            "case_ref": {"type": "STRING"},
                            "label": {"type": "STRING", "enum": ["0", "1", "2", "3", "ABSTAIN"]},
                        },
                        "required": ["case_ref", "label"],
                    },
                },
            },
            "required": ["cases"],
        },
    }


def parse_progress_function_call(raw_response: str, expected_case_ids, _expected_feasibility=None) -> dict[str, dict]:
    expected = [str(case_id) for case_id in expected_case_ids]
    if len(expected) != 1:
        raise LabelParseError("Progress Monitoring Gemma parser requires one case")
    try:
        data = json.loads(raw_response)
    except (TypeError, json.JSONDecodeError) as exc:
        raise LabelParseError("malformed Progress Monitoring Gemma response") from exc
    candidates = data.get("candidates") if isinstance(data, dict) else None
    parts = candidates[0].get("content", {}).get("parts", []) if isinstance(candidates, list) and candidates else []
    calls = [part.get("functionCall") for part in parts if isinstance(part, dict) and "functionCall" in part]
    if len(calls) != 1 or calls[0].get("name") != PROGRESS_FUNCTION_NAME:
        raise LabelParseError("response must contain one Progress Monitoring function call")
    args = calls[0].get("args")
    if not isinstance(args, dict) or set(args) != {"cases"} or not isinstance(args["cases"], list) or len(args["cases"]) != 1:
        raise LabelParseError("Progress Monitoring function call must contain one case")
    case = args["cases"][0]
    if not isinstance(case, dict) or set(case) != {"case_ref", "label"} or case["case_ref"] != "C01":
        raise LabelParseError("Progress Monitoring case_ref must be C01")
    value = case["label"]
    if isinstance(value, str) and value in {"0", "1", "2", "3"}:
        value = int(value)
    elif type(value) is not int or value not in (0, 1, 2, 3):
        if value != "ABSTAIN":
            raise LabelParseError(f"invalid Progress Monitoring label: {value!r}")
    return {expected[0]: {"case_id": expected[0], "prompt_version": A4_PROGRESS_GEMMA_PROMPT_VERSION,
                         "labels": {"A4": {"label": value}}}}
