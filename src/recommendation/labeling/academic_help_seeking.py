"""Gemma single-case contract for the A4 Academic Help-Seeking candidate."""

from __future__ import annotations

import json
from collections.abc import Mapping

from .constants import A4_ACADEMIC_HELP_GEMMA_PROMPT_VERSION
from .parser import LabelParseError

ACADEMIC_HELP_STATE_FIELDS = (
    "stage", "risk_probability", "inactive_streak", "active_days_ratio",
    "recent_activity", "activity_trend", "assessment_completion", "missing_assessments",
)
ACADEMIC_HELP_FUNCTION_NAME = "submit_academic_help_seeking_label"


def build_academic_help_prompt(case_payload: Mapping) -> str:
    state = {field: case_payload[field] for field in ACADEMIC_HELP_STATE_FIELDS}
    return f"""You are producing one weak relevance label for the A4 Academic Help-Seeking candidate.
Prompt version: {A4_ACADEMIC_HELP_GEMMA_PROMPT_VERSION}.

Definition: Academic Help-Seeking encourages the learner to seek appropriate academic assistance from
an instructor, tutor, advisor, or legitimate academic support when observable learning difficulties
indicate that additional help may be useful.

Keep this distinct from A1 Assessment Recovery (recover assessments), A2 Re-engagement (return to
learning interaction), A3 Study Planning (organize future study), and A5 Retrieval Practice (quiz or
self-test recall). Do not infer whether help is available, whether the learner already asked for help,
or any personal circumstances.

Use only the supplied evidence. Do not invent facts, infer future information, use outcomes, or return
prose. Return exactly one function call with case_ref `C01` and one label.

Rubric: 0=NOT_RELEVANT, 1=SLIGHTLY_RELEVANT, 2=RELEVANT, 3=HIGHLY_RELEVANT,
ABSTAIN=insufficient observable evidence.

Student State (case reference is an alias):
{json.dumps(state, ensure_ascii=False, sort_keys=True)}
"""


def academic_help_function_declaration() -> dict:
    return {
        "name": ACADEMIC_HELP_FUNCTION_NAME,
        "description": "Submit exactly one Academic Help-Seeking relevance label.",
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


def parse_academic_help_function_call(raw_response: str, expected_case_ids, _expected_feasibility=None) -> dict[str, dict]:
    expected = [str(case_id) for case_id in expected_case_ids]
    if len(expected) != 1:
        raise LabelParseError("Academic Help-Seeking Gemma parser requires one case")
    try:
        data = json.loads(raw_response)
    except (TypeError, json.JSONDecodeError) as exc:
        raise LabelParseError("malformed Academic Help-Seeking Gemma response") from exc
    candidates = data.get("candidates") if isinstance(data, dict) else None
    parts = candidates[0].get("content", {}).get("parts", []) if isinstance(candidates, list) and candidates else []
    calls = [part.get("functionCall") for part in parts if isinstance(part, dict) and "functionCall" in part]
    if len(calls) != 1 or calls[0].get("name") != ACADEMIC_HELP_FUNCTION_NAME:
        raise LabelParseError("response must contain one Academic Help-Seeking function call")
    args = calls[0].get("args")
    if not isinstance(args, dict) or set(args) != {"cases"} or not isinstance(args["cases"], list) or len(args["cases"]) != 1:
        raise LabelParseError("Academic Help-Seeking function call must contain one case")
    case = args["cases"][0]
    if not isinstance(case, dict) or set(case) != {"case_ref", "label"} or case["case_ref"] != "C01":
        raise LabelParseError("Academic Help-Seeking case_ref must be C01")
    value = case["label"]
    if isinstance(value, str) and value in {"0", "1", "2", "3"}:
        value = int(value)
    elif type(value) is not int or value not in (0, 1, 2, 3):
        if value != "ABSTAIN":
            raise LabelParseError(f"invalid Academic Help-Seeking label: {value!r}")
    return {expected[0]: {"case_id": expected[0], "prompt_version": A4_ACADEMIC_HELP_GEMMA_PROMPT_VERSION,
                         "labels": {"A4": {"label": value}}}}
