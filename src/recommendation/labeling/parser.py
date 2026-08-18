"""Strict offline parser and validator for provider responses."""

from __future__ import annotations

import json
from collections.abc import Mapping

from .constants import ACTION_IDS, ABSTAIN_REASONS, CONFIDENCE_VALUES, LABEL_VALUES, PROMPT_VERSION
from .panel_b_reference import PANEL_B_REFERENCE_PROMPT_VERSION

ALLOWED_PROMPT_VERSIONS = {PROMPT_VERSION, PANEL_B_REFERENCE_PROMPT_VERSION}


class LabelParseError(ValueError):
    pass


def parse_llm_response(raw_text: str, expected_case_ids, expected_feasibility=None, *, prompt_version=PROMPT_VERSION) -> dict[str, dict]:
    if prompt_version not in ALLOWED_PROMPT_VERSIONS:
        raise LabelParseError(f"unsupported prompt_version: {prompt_version}")
    if not isinstance(raw_text, str):
        raise LabelParseError("response is not text")
    try:
        data = json.loads(raw_text)
    except json.JSONDecodeError as exc:
        raise LabelParseError(f"malformed JSON: {exc.msg}") from exc
    if isinstance(data, Mapping) and "results" in data:
        results = data["results"]
    elif isinstance(data, Mapping):
        results = [data]
    else:
        raise LabelParseError("top-level response must be an object")
    if not isinstance(results, list):
        raise LabelParseError("results must be a list")
    expected = [str(x) for x in expected_case_ids]
    if len(set(expected)) != len(expected):
        raise LabelParseError("expected case IDs contain duplicates")
    output: dict[str, dict] = {}
    for result in results:
        if not isinstance(result, Mapping):
            raise LabelParseError("each result must be an object")
        if set(result) - {"case_id", "prompt_version", "labels"}:
            raise LabelParseError("unexpected result field")
        case_id = str(result.get("case_id", ""))
        if case_id not in expected:
            raise LabelParseError(f"unexpected case_id: {case_id}")
        if case_id in output:
            raise LabelParseError(f"duplicate case_id: {case_id}")
        if result.get("prompt_version") != prompt_version:
            raise LabelParseError(f"invalid prompt_version for {case_id}")
        labels = result.get("labels")
        if not isinstance(labels, Mapping) or set(labels) != set(ACTION_IDS):
            raise LabelParseError(f"labels must contain exactly A1-A5 for {case_id}")
        normalized = {}
        for action_id in ACTION_IDS:
            item = labels[action_id]
            if not isinstance(item, Mapping) or "label" not in item:
                raise LabelParseError(f"missing label for {case_id}/{action_id}")
            if set(item) - {"label", "confidence", "reason"}:
                raise LabelParseError(f"unexpected label field for {case_id}/{action_id}")
            label = item["label"]
            if type(label) is int and label in (0, 1, 2, 3):
                normalized_label = label
            elif isinstance(label, str) and label in {"0", "1", "2", "3"}:
                normalized_label = int(label)
            elif label == "ABSTAIN":
                normalized_label = label
            else:
                raise LabelParseError(f"invalid label for {case_id}/{action_id}")
            if "confidence" in item and item["confidence"] not in CONFIDENCE_VALUES:
                raise LabelParseError(f"invalid confidence for {case_id}/{action_id}")
            feasibility = (expected_feasibility or {}).get(case_id, {}).get(action_id)
            reason = item.get("reason")
            if normalized_label == "ABSTAIN":
                if reason is not None and reason not in ABSTAIN_REASONS:
                    raise LabelParseError(f"ABSTAIN requires a valid reason for {case_id}/{action_id}")
                if feasibility == "INFEASIBLE" and reason is not None and reason != "INFEASIBLE":
                    raise LabelParseError(f"infeasible action must use INFEASIBLE reason for {case_id}/{action_id}")
                if reason == "INFEASIBLE" and feasibility is not None and feasibility != "INFEASIBLE":
                    raise LabelParseError(f"INFEASIBLE reason is invalid for {feasibility} action {case_id}/{action_id}")
            elif "reason" in item:
                raise LabelParseError(f"numeric label cannot include reason for {case_id}/{action_id}")
            if feasibility == "INFEASIBLE" and normalized_label != "ABSTAIN":
                raise LabelParseError(f"infeasible action must be ABSTAIN for {case_id}/{action_id}")
            normalized_item = dict(item)
            normalized_item["label"] = normalized_label
            normalized[action_id] = normalized_item
        output[case_id] = {"case_id": case_id, "prompt_version": prompt_version, "labels": normalized}
    if set(output) != set(expected):
        raise LabelParseError(f"missing case IDs: {sorted(set(expected) - set(output))}")
    return output
