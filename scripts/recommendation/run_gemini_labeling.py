"""Run Gemini jobs with GEMINI_API_KEY supplied only by the user environment."""

from __future__ import annotations

import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.recommendation._run_labeling import common_parser, execute, gemini_request
from src.recommendation.labeling.a4_replacement import parse_replacement_response
from src.recommendation.labeling.constants import A4_PROGRESS_GEMINI31_PROMPT_VERSION, A4_REPLACEMENT_PROMPT_VERSION
from src.recommendation.labeling.panel_b_reference import PANEL_B_REFERENCE_PROMPT_VERSION
from src.recommendation.labeling.parser import parse_llm_response
from src.recommendation.labeling.progress_monitoring_gemini31 import parse_progress_monitoring_gemini31_response
from src.recommendation.labeling.runtime import load_jsonl


if __name__ == "__main__":
    parser = common_parser("Run Gemini labeling jobs")
    args = parser.parse_args()
    jobs = load_jsonl(args.input)
    prompt_versions = {str(job.get("prompt_version")) for job in jobs}
    if A4_PROGRESS_GEMINI31_PROMPT_VERSION in prompt_versions:
        if prompt_versions != {A4_PROGRESS_GEMINI31_PROMPT_VERSION}:
            raise ValueError("cannot mix Progress Monitoring Gemini 3.1 jobs with other Gemini jobs")
        execute(args, gemini_request, response_parser=parse_progress_monitoring_gemini31_response)
    elif A4_REPLACEMENT_PROMPT_VERSION in prompt_versions:
        if prompt_versions != {A4_REPLACEMENT_PROMPT_VERSION}:
            raise ValueError("cannot mix A4 replacement jobs with standard Gemini jobs")
        execute(args, gemini_request, response_parser=parse_replacement_response)
    elif PANEL_B_REFERENCE_PROMPT_VERSION in prompt_versions:
        if prompt_versions != {PANEL_B_REFERENCE_PROMPT_VERSION}:
            raise ValueError("cannot mix Panel B reference jobs with other Gemini jobs")
        execute(
            args,
            gemini_request,
            response_parser=lambda raw, ids, feas=None: parse_llm_response(
                raw, ids, feas, prompt_version=PANEL_B_REFERENCE_PROMPT_VERSION
            ),
        )
    else:
        execute(args, gemini_request)
