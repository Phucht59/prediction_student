"""Panel B automated reference jobs. No model scores, no future fields, no API calls."""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping

from .constants import RUBRIC
from .payload import ALLOWED_FIELDS, FORBIDDEN_FIELDS

PANEL_B_REFERENCE_PROMPT_VERSION = "recommendation_panel_b_reference_v1"
PANEL_B_ACTIONS = ("A1", "A2", "A3", "A4", "A5")
PANEL_B_ACTION_DEFINITIONS = {
    "A1": "Assessment Recovery: prioritize completing or recovering missing or incomplete assessments.",
    "A2": "Re-engagement: encourage returning to interaction with the learning environment when engagement has reduced or stopped.",
    "A3": "Study Planning: improve study rhythm and organize a more regular learning plan.",
    "A4": "Progress Monitoring: review currently observed learning progress and whether the learner is on track. This is not Content Review and does not require content availability.",
    "A5": "Retrieval Practice: practice recalling knowledge through quizzes, self-tests, or retrieval activities.",
}


def build_panel_b_payload(row: Mapping, feasibility_statuses: Mapping[str, str]) -> dict:
    payload = {"case_id": str(row["case_id"]), "prompt_version": PANEL_B_REFERENCE_PROMPT_VERSION}
    def _json_value(value):
        return value.item() if hasattr(value, "item") else value
    payload.update({field: _json_value(row[field]) for field in ALLOWED_FIELDS})
    payload["actions"] = list(PANEL_B_ACTIONS)
    payload["feasibility"] = {action_id: str(feasibility_statuses[action_id]) for action_id in PANEL_B_ACTIONS}
    payload["rubric"] = dict(RUBRIC)
    leaked = FORBIDDEN_FIELDS.intersection(payload)
    if leaked:
        raise ValueError(f"Panel B reference payload contains forbidden fields: {sorted(leaked)}")
    for banned in ("raw_score", "relevance_score", "rank", "top_positive_reasons", "model_version", "final_result"):
        if banned in payload:
            raise ValueError("Panel B reference payload must not include model output")
    return payload


def build_panel_b_prompt(cases: Iterable[Mapping]) -> str:
    case_text = json.dumps(list(cases), ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    actions = "\n".join(f"- {key}: {PANEL_B_ACTION_DEFINITIONS[key]}" for key in PANEL_B_ACTIONS)
    rubric = "\n".join(f"- {key}: {value}" for key, value in RUBRIC.items())
    output = {"results": [{"case_id": "case_id_from_input", "prompt_version": PANEL_B_REFERENCE_PROMPT_VERSION,
                            "labels": {action_id: {"label": "0|1|2|3|ABSTAIN", "confidence": "LOW|MEDIUM|HIGH"}
                                        for action_id in PANEL_B_ACTIONS}}]}
    return f"""You are producing automated reference relevance judgments for a sealed holdout student-learning state.
Prompt version: {PANEL_B_REFERENCE_PROMPT_VERSION}.

Use only the supplied fields. Do not invent missing facts, infer future behavior, use outcomes,
or infer information after the supplied stage. Evaluate each action independently. Feasibility is
separate from relevance. If an action has feasibility status INFEASIBLE, output label ABSTAIN and
reason INFEASIBLE for that action; never convert that status to numeric label 0.

Actions:
{actions}

Relevance rubric:
{rubric}

Return strict JSON only. Do not return markdown, explanations, a plan, or recommendation prose.
Return exactly one result per input case, with exactly A1-A5 labels.

Required output shape:
{json.dumps(output, ensure_ascii=False, indent=2)}

Input cases:
{case_text}
"""


ALLOWED_REFERENCE_LABELS = {0, 1, 2, 3, "0", "1", "2", "3", "ABSTAIN"}


def canonicalize_reference_label(value):
    if value in {0, 1, 2, 3}:
        return int(value)
    if value in {"0", "1", "2", "3"}:
        return int(value)
    if value == "ABSTAIN":
        return "ABSTAIN"
    raise ValueError(f"invalid Panel B reference label: {value!r}")


def numeric_reference_label(value) -> int | None:
    label = canonicalize_reference_label(value)
    return None if label == "ABSTAIN" else int(label)


def normalize_dual_reference(label_35, label_31) -> dict:
    left, right = numeric_reference_label(label_35), numeric_reference_label(label_31)
    if left is not None and right is not None:
        return {"reference_relevance": (left + right) / 2.0, "reference_status": "DUAL_SOURCE"}
    if left is not None:
        return {"reference_relevance": float(left), "reference_status": "SINGLE_SOURCE"}
    if right is not None:
        return {"reference_relevance": float(right), "reference_status": "SINGLE_SOURCE"}
    return {"reference_relevance": None, "reference_status": "NO_REFERENCE"}


def _extract_label(item) -> object:
    if isinstance(item, dict):
        if "label" not in item:
            raise ValueError("parsed label object is missing label")
        return item["label"]
    return item


def load_raw_panel_b_labels(path, *, expected_model: str | None = None) -> dict[str, dict]:
    from pathlib import Path

    records = [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(records) != 150:
        raise ValueError(f"{path} must contain exactly 150 records, found {len(records)}")
    by_case = {}
    for record in records:
        if record.get("status") != "completed":
            raise ValueError(f"{path} contains a non-completed record: {record.get('case_id')}")
        if record.get("prompt_version") != PANEL_B_REFERENCE_PROMPT_VERSION:
            raise ValueError(f"{path} has unexpected prompt_version")
        if expected_model is not None and record.get("model") != expected_model:
            raise ValueError(f"{path} model {record.get('model')!r} != {expected_model!r}")
        case_id = str(record["case_id"])
        parsed = record.get("parsed_labels")
        if not isinstance(parsed, dict) or "labels" not in parsed:
            raise ValueError(f"{path} record {case_id} is missing parsed_labels.labels")
        labels = parsed["labels"]
        if set(labels) != set(PANEL_B_ACTIONS):
            raise ValueError(f"{path} record {case_id} does not contain exactly A1-A5")
        if case_id in by_case:
            raise ValueError(f"duplicate case_id in {path}: {case_id}")
        by_case[case_id] = {
            "case_id": case_id,
            "labels": {action: canonicalize_reference_label(_extract_label(labels[action])) for action in PANEL_B_ACTIONS},
            "model": record.get("model"),
            "provider": record.get("provider"),
            "prompt_version": record.get("prompt_version"),
            "job_id": record.get("job_id"),
        }
    if len(by_case) != 150:
        raise ValueError(f"{path} must cover 150 unique cases")
    return by_case


def build_panel_b_reference_table(gemini35_path, gemini31_path, panel_b_ids: set[str]):
    from ..feasibility.rules_v2 import LEGACY_TO_FINAL

    left = load_raw_panel_b_labels(gemini35_path, expected_model="gemini-3.5-flash-lite")
    right = load_raw_panel_b_labels(gemini31_path, expected_model="gemini-3.1-flash-lite")
    if set(left) != panel_b_ids or set(right) != panel_b_ids:
        raise ValueError("Panel B reference case IDs do not match the sealed 150-case holdout")
    rows = []
    for case_id in sorted(panel_b_ids):
        for action_key in PANEL_B_ACTIONS:
            label_35 = left[case_id]["labels"][action_key]
            label_31 = right[case_id]["labels"][action_key]
            aggregated = normalize_dual_reference(label_35, label_31)
            if aggregated["reference_status"] == "NO_REFERENCE" and aggregated["reference_relevance"] is not None:
                raise ValueError("NO_REFERENCE must not carry a fabricated relevance")
            if aggregated["reference_status"] == "NO_REFERENCE" and (label_35 == 0 or label_31 == 0):
                raise ValueError("ABSTAIN must not be mapped to 0")
            rows.append({
                "case_id": case_id,
                "action_key": action_key,
                "action_id": LEGACY_TO_FINAL[action_key],
                "label_gemini35": "ABSTAIN" if label_35 == "ABSTAIN" else str(int(label_35)),
                "label_gemini31": "ABSTAIN" if label_31 == "ABSTAIN" else str(int(label_31)),
                "reference_relevance": aggregated["reference_relevance"],
                "reference_status": aggregated["reference_status"],
                "model_gemini35": left[case_id]["model"],
                "model_gemini31": right[case_id]["model"],
                "provider_gemini35": left[case_id]["provider"],
                "provider_gemini31": right[case_id]["provider"],
                "prompt_version_gemini35": left[case_id]["prompt_version"],
                "prompt_version_gemini31": right[case_id]["prompt_version"],
                "job_id_gemini35": left[case_id]["job_id"],
                "job_id_gemini31": right[case_id]["job_id"],
            })
    frame = __import__("pandas").DataFrame(rows).sort_values(["case_id", "action_key"]).reset_index(drop=True)
    if len(frame) != 750 or frame.duplicated(["case_id", "action_id"]).any():
        raise ValueError("normalized Panel B reference must contain 750 unique case-action rows")
    if frame["reference_status"].eq("NO_REFERENCE").any() and frame.loc[frame["reference_status"].eq("NO_REFERENCE"), "reference_relevance"].notna().any():
        raise ValueError("NO_REFERENCE rows must not fabricate reference_relevance")
    return frame


def pairwise_reference_agreement(frame) -> dict:
    import math

    from sklearn.metrics import cohen_kappa_score

    report = {}
    for action_id, group in frame.groupby("action_id"):
        left = group["label_gemini35"].tolist()
        right = group["label_gemini31"].tolist()
        exact = sum(a == b for a, b in zip(left, right))
        numeric = [(numeric_reference_label(a), numeric_reference_label(b)) for a, b in zip(left, right)]
        pairs = [(a, b) for a, b in numeric if a is not None and b is not None]
        def _kappa(weights):
            if len(pairs) < 2:
                return None
            y1, y2 = zip(*pairs)
            if len(set(y1)) < 2 and len(set(y2)) < 2:
                return None
            value = float(cohen_kappa_score(y1, y2, labels=[0, 1, 2, 3], weights=weights))
            return None if math.isnan(value) else value
        report[action_id] = {
            "n": int(len(group)),
            "exact_agreement": int(exact),
            "exact_agreement_rate": float(exact / len(group)),
            "numeric_overlap": len(pairs),
            "linear_weighted_kappa": _kappa("linear"),
            "quadratic_weighted_kappa": _kappa("quadratic"),
            "gemini35_distribution": {key: int((group["label_gemini35"].astype(str) == key).sum()) for key in ("0", "1", "2", "3", "ABSTAIN")},
            "gemini31_distribution": {key: int((group["label_gemini31"].astype(str) == key).sum()) for key in ("0", "1", "2", "3", "ABSTAIN")},
            "reference_status_counts": group["reference_status"].value_counts().astype(int).to_dict(),
        }
    left = frame["label_gemini35"].tolist()
    right = frame["label_gemini31"].tolist()
    exact = sum(a == b for a, b in zip(left, right))
    report["_overall"] = {
        "n": int(len(frame)),
        "exact_agreement": int(exact),
        "exact_agreement_rate": float(exact / len(frame)),
        "same_family_warning": "Gemini 3.5 and Gemini 3.1 are the same model family and are not independent expert annotators.",
    }
    return report

