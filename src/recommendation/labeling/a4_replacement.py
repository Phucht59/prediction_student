"""Prompt and strict parser for the offline-prepared A4 replacement study."""

from __future__ import annotations

import json
from collections.abc import Mapping

from .constants import A4_REPLACEMENT_PROMPT_VERSION
from .parser import LabelParseError

REPLACEMENT_ACTIONS = ("B1_PROGRESS_MONITORING", "B2_ACADEMIC_HELP_SEEKING")
REPLACEMENT_LABELS = {"0", "1", "2", "3", "ABSTAIN"}
REPLACEMENT_STATE_FIELDS = (
    "stage", "risk_probability", "risk_band", "inactive_streak", "active_days_ratio",
    "recent_activity", "activity_trend", "assessment_completion", "missing_assessments",
    "course_progress", "quiz_activity", "vle_available",
)

B1_DEFINITION = (
    "Progress Monitoring: encourage the learner to explicitly review current learning progress, "
    "identify whether they are falling behind, and monitor improvement over time."
)
B2_DEFINITION = (
    "Academic Help Seeking: encourage the learner to seek appropriate academic assistance from an "
    "instructor, tutor, advisor, or other legitimate academic support when current difficulties "
    "indicate that additional support may be useful. Do not assume that the learner has already sought help."
)


def build_replacement_prompt(cases) -> str:
    case_list = list(cases)
    output_shape = {
        "case_id": "anonymous_case_id_from_input",
        "labels": {
            "B1_PROGRESS_MONITORING": {"label": "0|1|2|3|ABSTAIN"},
            "B2_ACADEMIC_HELP_SEEKING": {"label": "0|1|2|3|ABSTAIN"},
        },
    }
    return f"""You are producing an offline weak relevance comparison for two candidate replacements for unsupported A4 Content Review.
Prompt version: {A4_REPLACEMENT_PROMPT_VERSION}.

Use only the supplied Student State fields. Do not invent features, infer future information,
use outcomes, or infer information after the supplied stage. Evaluate B1 and B2 independently.
Do not force one candidate to be better than the other. High risk alone must not imply label 3.
Do not assume that the learner has already sought help.

Candidate actions:
- B1_PROGRESS_MONITORING: {B1_DEFINITION}
- B2_ACADEMIC_HELP_SEEKING: {B2_DEFINITION}

Important distinction: B1 inspects and monitors current progress or gaps; it is not A3 Study Planning,
which organizes a future study schedule or rhythm. B2 is academic assistance seeking, not a claim that
help has already been sought.

Rubric:
- 0 = NOT_RELEVANT
- 1 = SLIGHTLY_RELEVANT
- 2 = RELEVANT
- 3 = HIGHLY_RELEVANT
- ABSTAIN = insufficient observable evidence

Return strict JSON only. No markdown, explanation, recommendation prose, or invented data.
For a batch, return {{"results": [one object per input case]}}. Each result must have exactly the
following fields and exactly the two labels shown:
{json.dumps(output_shape, ensure_ascii=False, indent=2)}

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
    raise LabelParseError(f"invalid replacement label: {value!r}")


def parse_replacement_response(raw_text: str, expected_case_ids, _feasibility=None) -> dict[str, dict]:
    """Parse only the B1/B2 JSON contract; no A1-A5 parser is reused."""
    try:
        data = json.loads(raw_text)
    except (TypeError, json.JSONDecodeError) as exc:
        raise LabelParseError("replacement response is not valid JSON") from exc
    if isinstance(data, Mapping) and set(data) == {"results"}:
        results = data["results"]
    elif isinstance(data, Mapping):
        results = [data]
    else:
        raise LabelParseError("replacement response must be an object or results object")
    expected = [str(case_id) for case_id in expected_case_ids]
    if len(expected) != len(set(expected)):
        raise LabelParseError("expected replacement case IDs contain duplicates")
    if not isinstance(results, list):
        raise LabelParseError("replacement results must be a list")
    parsed: dict[str, dict] = {}
    for result in results:
        if not isinstance(result, Mapping) or set(result) != {"case_id", "labels"}:
            raise LabelParseError("each replacement result must contain exactly case_id and labels")
        case_id = str(result["case_id"])
        if case_id not in expected:
            raise LabelParseError(f"unexpected replacement case_id: {case_id}")
        if case_id in parsed:
            raise LabelParseError(f"duplicate replacement case_id: {case_id}")
        labels = result["labels"]
        if not isinstance(labels, Mapping) or set(labels) != set(REPLACEMENT_ACTIONS):
            raise LabelParseError(f"replacement labels must contain exactly B1 and B2 for {case_id}")
        normalized = {}
        for action_id in REPLACEMENT_ACTIONS:
            item = labels[action_id]
            if not isinstance(item, Mapping) or set(item) != {"label"}:
                raise LabelParseError(f"replacement label object must contain only label for {case_id}/{action_id}")
            normalized[action_id] = {"label": _normalize_label(item["label"])}
        parsed[case_id] = {"case_id": case_id, "labels": normalized}
    if set(parsed) != set(expected):
        raise LabelParseError(f"missing replacement case IDs: {sorted(set(expected) - set(parsed))}")
    return parsed
