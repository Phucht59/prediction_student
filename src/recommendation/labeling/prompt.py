"""Shared provider-neutral prompt for weak relevance judgments."""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping

from .constants import ACTION_DEFINITIONS, ACTION_IDS, PROMPT_VERSION, PROMPT_VERSION_B, RUBRIC


def build_prompt(cases: Iterable[Mapping]) -> str:
    case_text = json.dumps(list(cases), ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    actions = "\n".join(f"- {key}: {ACTION_DEFINITIONS[key]}" for key in ACTION_IDS)
    rubric = "\n".join(f"- {key}: {value}" for key, value in RUBRIC.items())
    output = {"results": [{"case_id": "case_id_from_input", "prompt_version": PROMPT_VERSION,
                            "labels": {action_id: {"label": "0|1|2|3|ABSTAIN", "confidence": "LOW|MEDIUM|HIGH"}
                                        for action_id in ACTION_IDS}}]}
    return f"""You are producing weak relevance judgments for a student-learning state.
Prompt version: {PROMPT_VERSION}.

Use only the supplied fields. Do not invent missing facts, infer future behavior, use outcomes,
or infer information after the supplied stage. Evaluate each action independently. Feasibility is
separate from relevance. If an action has feasibility status INFEASIBLE, output label ABSTAIN and
reason INFEASIBLE for that action; never convert that status to numeric label 0. For other actions,
use ABSTAIN with reason INSUFFICIENT_INFORMATION only when the supplied state is insufficient.

Actions:
{actions}

Relevance rubric:
{rubric}

Return strict JSON only. Do not return markdown, explanations, a plan, or recommendation prose.
Return exactly one result per input case, with exactly A1-A5 labels. Each label object may include
auxiliary confidence LOW, MEDIUM, or HIGH; confidence is not a weight. An ABSTAIN label must
include reason INFEASIBLE or INSUFFICIENT_INFORMATION.

Required output shape:
{json.dumps(output, ensure_ascii=False, indent=2)}

Input cases:
{case_text}
"""


def build_prompt_v1b(cases: Iterable[Mapping]) -> str:
    """Prompt-robustness variant: same fields/actions/rubric/output schema, tighter wording only."""
    prompt = build_prompt(cases).replace(PROMPT_VERSION, PROMPT_VERSION_B)
    tightening = """

Additional robustness instructions:
- Use only supplied evidence.
- Do not infer unavailable features.
- Evaluate each action independently.
- High risk alone must not increase all action scores.
- UNKNOWN is not INFEASIBLE.
- ABSTAIN only when current evidence is insufficient.
- Do not force a numeric label when evidence is absent.
- Do not generate recommendation prose.
"""
    return prompt + tightening
