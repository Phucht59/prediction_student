"""Run Gemma jobs against a user-controlled local OpenAI-compatible endpoint."""

from __future__ import annotations

import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.recommendation._run_labeling import (
    academic_help_gemma_request, common_parser, execute, gemma_request, parse_gemma_function_call,
    progress_monitoring_gemma_request,
)
from src.recommendation.labeling.constants import A4_ACADEMIC_HELP_GEMMA_PROMPT_VERSION, A4_PROGRESS_GEMMA_PROMPT_VERSION
from src.recommendation.labeling.academic_help_seeking import parse_academic_help_function_call
from src.recommendation.labeling.progress_monitoring import parse_progress_function_call
from src.recommendation.labeling.runtime import load_jsonl


if __name__ == "__main__":
    parser = common_parser("Run local Gemma labeling jobs", default_rpm=27.0)
    args = parser.parse_args()
    prompt_versions = {str(job.get("prompt_version")) for job in load_jsonl(args.input)}
    if A4_ACADEMIC_HELP_GEMMA_PROMPT_VERSION in prompt_versions:
        if prompt_versions != {A4_ACADEMIC_HELP_GEMMA_PROMPT_VERSION}:
            raise ValueError("cannot mix Academic Help-Seeking Gemma jobs with other Gemma jobs")
        execute(args, academic_help_gemma_request, response_parser=parse_academic_help_function_call)
    elif A4_PROGRESS_GEMMA_PROMPT_VERSION in prompt_versions:
        if prompt_versions != {A4_PROGRESS_GEMMA_PROMPT_VERSION}:
            raise ValueError("cannot mix Progress Monitoring Gemma jobs with standard Gemma jobs")
        execute(args, progress_monitoring_gemma_request, response_parser=parse_progress_function_call)
    else:
        execute(args, gemma_request, response_parser=parse_gemma_function_call)
