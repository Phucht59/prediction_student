"""Shared CLI plumbing for the two local-only provider runners."""

from __future__ import annotations

import argparse
import json
import os
import socket
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.recommendation.labeling.constants import PROMPT_VERSION  # noqa: E402
from src.recommendation.labeling.progress_monitoring import (  # noqa: E402
    PROGRESS_FUNCTION_NAME, build_progress_monitoring_prompt, parse_progress_function_call,
    progress_function_declaration,
)
from src.recommendation.labeling.academic_help_seeking import (  # noqa: E402
    ACADEMIC_HELP_FUNCTION_NAME, academic_help_function_declaration,
    build_academic_help_prompt,
)
from src.recommendation.labeling.parser import LabelParseError, parse_llm_response  # noqa: E402
from src.recommendation.labeling.runtime import TransientAPIError, load_jsonl, run_jobs  # noqa: E402

GEMMA_FUNCTION_NAME = "submit_relevance_labels"
_LABEL_ENUM = ["0", "1", "2", "3", "ABSTAIN"]
_REASON_ENUM = ["INFEASIBLE", "INSUFFICIENT_INFORMATION", "NONE"]


def build_gemma_alias_map(case_ids) -> dict[str, str]:
    return {f"C{index:02d}": str(case_id) for index, case_id in enumerate(case_ids, start=1)}


def build_gemma_prompt(job: dict) -> str:
    """Replace real IDs in the Gemma-facing prompt with batch-local aliases."""
    alias_map = build_gemma_alias_map(job["case_ids"])
    prompt = job["prompt"]
    for alias, real_case_id in alias_map.items():
        prompt = prompt.replace(real_case_id, alias)
    prompt = prompt.replace('"case_id"', '"case_ref"')
    return f"""Gemma case-reference contract:
The supplied case references are aliases C01 through C{len(alias_map):02d}. Return those exact aliases in the function call as `case_ref`; never return or invent a real case ID.
UNKNOWN does not mean INFEASIBLE.
Do not infer unavailable features.
High risk does not imply all actions are highly relevant.
Evaluate each action independently.
If action feasibility is INFEASIBLE, return ABSTAIN with reason INFEASIBLE.
If evidence is insufficient but the action is not infeasible, return ABSTAIN with reason INSUFFICIENT_INFORMATION.

{prompt}
"""


def _gemma_function_declaration() -> dict:
    label_schema = {
        "type": "OBJECT",
        "properties": {
            "label": {"type": "STRING", "enum": _LABEL_ENUM},
            "reason": {"type": "STRING", "enum": _REASON_ENUM},
        },
        "required": ["label", "reason"],
    }
    return {
        "name": GEMMA_FUNCTION_NAME,
        "description": "Submit relevance labels for every supplied student-state case and action.",
        "parameters": {
            "type": "OBJECT",
            "properties": {
                "cases": {
                    "type": "ARRAY",
                    "items": {
                        "type": "OBJECT",
                        "properties": {
                            "case_ref": {"type": "STRING"},
                            "labels": {
                                "type": "OBJECT",
                                "properties": {action_id: label_schema for action_id in ("A1", "A2", "A3", "A4", "A5")},
                                "required": ["A1", "A2", "A3", "A4", "A5"],
                            },
                        },
                        "required": ["case_ref", "labels"],
                    },
                },
            },
            "required": ["cases"],
        },
    }


def parse_gemma_function_call(raw_response: str, expected_case_ids, expected_feasibility=None) -> dict[str, dict]:
    """Parse only ``functionCall.args``; prose/text parts are never used for labels."""
    expected_case_ids = list(expected_case_ids)
    if len(expected_case_ids) != 1:
        raise LabelParseError("Gemma single-case parser requires exactly one expected case")
    try:
        data = json.loads(raw_response)
    except json.JSONDecodeError as exc:
        raise LabelParseError(f"malformed API response JSON: {exc.msg}") from exc
    if not isinstance(data, dict):
        raise LabelParseError("Gemma API response must be an object")
    candidates = data.get("candidates")
    if not isinstance(candidates, list) or not candidates or not isinstance(candidates[0], dict):
        raise LabelParseError("Gemma response has no candidate content")
    content = candidates[0].get("content")
    if not isinstance(content, dict):
        raise LabelParseError("Gemma response has no candidate content")
    parts = content.get("parts", [])
    if not isinstance(parts, list):
        raise LabelParseError("Gemma candidate parts must be a list")
    calls = [part.get("functionCall") for part in parts if isinstance(part, dict) and "functionCall" in part]
    if len(calls) != 1:
        raise LabelParseError("Gemma response must contain exactly one functionCall")
    function_call = calls[0]
    if not isinstance(function_call, dict) or function_call.get("name") != GEMMA_FUNCTION_NAME:
        raise LabelParseError("unexpected Gemma function name")
    args = function_call.get("args")
    if not isinstance(args, dict) or set(args) != {"cases"} or not isinstance(args["cases"], list):
        raise LabelParseError("functionCall.args must contain only cases")
    if len(args["cases"]) != 1:
        raise LabelParseError("Gemma function call must contain exactly one case")
    alias_map = build_gemma_alias_map(expected_case_ids)
    seen_aliases = set()
    results = []
    for case in args["cases"]:
        if not isinstance(case, dict) or set(case) != {"case_ref", "labels"}:
            raise LabelParseError("each function-call case must contain case_ref and labels")
        case_ref = case["case_ref"]
        if case_ref not in alias_map:
            raise LabelParseError(f"unknown case alias: {case_ref}")
        if case_ref in seen_aliases:
            raise LabelParseError(f"duplicate case alias: {case_ref}")
        seen_aliases.add(case_ref)
        labels = case["labels"]
        if not isinstance(labels, dict):
            raise LabelParseError("function-call labels must be an object")
        for action_id in ("A1", "A2", "A3", "A4", "A5"):
            if not isinstance(labels.get(action_id), dict) or "reason" not in labels[action_id]:
                raise LabelParseError(f"function-call label missing reason for {case_ref}/{action_id}")
        normalized_labels = {}
        for action_id, item in labels.items():
            normalized_item = dict(item)
            if normalized_item.get("reason") == "NONE" and normalized_item.get("label") != "ABSTAIN":
                normalized_item.pop("reason")
            normalized_labels[action_id] = normalized_item
        results.append({"case_id": alias_map[case_ref], "prompt_version": PROMPT_VERSION, "labels": normalized_labels})
    if seen_aliases != set(alias_map):
        missing = sorted(set(alias_map) - seen_aliases)
        raise LabelParseError(f"missing case aliases: {missing}")
    return parse_llm_response(json.dumps({"results": results}), expected_case_ids, expected_feasibility)


class APIRequestError(RuntimeError):
    """HTTP/network failure with diagnostics safe to persist in raw JSONL."""

    def __init__(self, message: str, *, status_code=None, response_body=None, timeout_type=None,
                 elapsed_seconds=None, transient=False):
        diagnostics = {
            "exception_class": self.__class__.__name__,
            "http_status": status_code,
            "response_body": response_body,
            "timeout_type": timeout_type,
            "elapsed_seconds": elapsed_seconds,
        }
        RuntimeError.__init__(self, message)
        self.diagnostics = diagnostics


class TransientAPIRequestError(TransientAPIError):
    """Transient HTTP/network failure eligible for the existing retry loop."""

    def __init__(self, message: str, **kwargs):
        diagnostics = {
            "exception_class": self.__class__.__name__,
            "http_status": kwargs.get("status_code"),
            "response_body": kwargs.get("response_body"),
            "timeout_type": kwargs.get("timeout_type"),
            "elapsed_seconds": kwargs.get("elapsed_seconds"),
        }
        TransientAPIError.__init__(self, message, diagnostics=diagnostics)


def _positive_rpm(value: str) -> float:
    try:
        rpm = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("--rpm-limit must be a number greater than zero") from exc
    if rpm <= 0:
        raise argparse.ArgumentTypeError("--rpm-limit must be greater than zero")
    return rpm


def _positive_timeout(value: str) -> float:
    try:
        timeout = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("--timeout-seconds must be greater than zero") from exc
    if timeout <= 0:
        raise argparse.ArgumentTypeError("--timeout-seconds must be greater than zero")
    return timeout


def common_parser(description: str, *, default_rpm: float = 12.0) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--batch-size", type=int, choices=(1, 5, 10), default=10)
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument("--retry-delay", type=float, default=2.0)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--rpm-limit", type=_positive_rpm, default=default_rpm,
                        help=f"maximum request rate; requests are evenly spaced (default: {default_rpm:g} RPM)")
    parser.add_argument("--timeout-seconds", type=_positive_timeout, default=300.0,
                        help="per-request timeout in seconds (default: 300)")
    return parser


def _post_json(url: str, body: dict, headers: dict[str, str], timeout: float = 300.0) -> dict:
    request = urllib.request.Request(url, data=json.dumps(body).encode("utf-8"), headers=headers, method="POST")
    started = time.monotonic()
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            response_body = response.read().decode("utf-8", errors="replace")
            try:
                return json.loads(response_body)
            except json.JSONDecodeError as exc:
                raise APIRequestError(
                    "response body is not valid JSON",
                    response_body=response_body,
                    elapsed_seconds=time.monotonic() - started,
                ) from exc
    except urllib.error.HTTPError as exc:
        response_body = exc.read().decode("utf-8", errors="replace")
        elapsed = time.monotonic() - started
        if exc.code == 429 or 500 <= exc.code < 600:
            raise TransientAPIRequestError(
                f"transient HTTP status {exc.code}", status_code=exc.code,
                response_body=response_body, elapsed_seconds=elapsed,
            ) from exc
        raise APIRequestError(
            f"HTTP status {exc.code}", status_code=exc.code,
            response_body=response_body, elapsed_seconds=elapsed,
        ) from exc
    except urllib.error.URLError as exc:
        elapsed = time.monotonic() - started
        reason = exc.reason
        is_timeout = isinstance(reason, (TimeoutError, socket.timeout)) or "timed out" in str(reason).lower()
        timeout_type = type(reason).__name__ if is_timeout else None
        error_type = "timeout" if is_timeout else "network error"
        raise TransientAPIRequestError(
            f"{error_type}: {reason}", timeout_type=timeout_type, elapsed_seconds=elapsed,
        ) from exc
    except (TimeoutError, socket.timeout) as exc:
        raise TransientAPIRequestError(
            f"timeout: {exc}", timeout_type=type(exc).__name__,
            elapsed_seconds=time.monotonic() - started,
        ) from exc


def _gemini_generate_content_request(job: dict, *, timeout_seconds: float) -> tuple[str, dict | None]:
    api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GEMMA_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY is required for the official Gemini generateContent endpoint")
    model = os.environ.get("GEMINI_MODEL", job["model"])
    endpoint = os.environ.get("GEMINI_BASE_URL", "https://generativelanguage.googleapis.com/v1beta")
    url = f"{endpoint.rstrip('/')}/models/{model}:generateContent"
    data = _post_json(url, {"contents": [{"parts": [{"text": job["prompt"]}]}],
                            "generationConfig": {"responseMimeType": "application/json"}},
                      {"Content-Type": "application/json", "x-goog-api-key": api_key}, timeout=timeout_seconds)
    try:
        text = data["candidates"][0]["content"]["parts"][0]["text"]
    except (KeyError, IndexError, TypeError) as exc:
        raise RuntimeError("Gemini generateContent response has no candidates[0].content.parts[0].text") from exc
    return str(text), data.get("usageMetadata")


def gemma_request(job: dict, *, timeout_seconds: float = 300.0) -> tuple[str, dict | None]:
    """Call the official Gemini generateContent endpoint with the Gemma job model."""
    api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GEMMA_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY is required for the official Gemini generateContent endpoint")
    model = os.environ.get("GEMMA_MODEL", job["model"])
    endpoint = os.environ.get("GEMINI_BASE_URL", "https://generativelanguage.googleapis.com/v1beta")
    url = f"{endpoint.rstrip('/')}/models/{model}:generateContent"
    body = {
        "contents": [{"parts": [{"text": build_gemma_prompt(job)}]}],
        "tools": [{"functionDeclarations": [_gemma_function_declaration()]}],
        "toolConfig": {"functionCallingConfig": {"mode": "ANY", "allowedFunctionNames": [GEMMA_FUNCTION_NAME]}},
    }
    data = _post_json(url, body, {"Content-Type": "application/json", "x-goog-api-key": api_key}, timeout=timeout_seconds)
    return json.dumps(data, ensure_ascii=False), data.get("usageMetadata")


def progress_monitoring_gemma_request(job: dict, *, timeout_seconds: float = 300.0) -> tuple[str, dict | None]:
    """Call the same official generateContent endpoint with the A4 schema."""
    api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GEMMA_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY is required for the official Gemini generateContent endpoint")
    model = os.environ.get("GEMMA_MODEL", job["model"])
    endpoint = os.environ.get("GEMINI_BASE_URL", "https://generativelanguage.googleapis.com/v1beta")
    payload = job["payload"][0]
    prompt = build_progress_monitoring_prompt(payload).replace(str(payload["case_id"]), "C01")
    body = {
        "contents": [{"parts": [{"text": prompt}]}],
        "tools": [{"functionDeclarations": [progress_function_declaration()]}],
        "toolConfig": {"functionCallingConfig": {"mode": "ANY", "allowedFunctionNames": [PROGRESS_FUNCTION_NAME]}},
    }
    data = _post_json(f"{endpoint.rstrip('/')}/models/{model}:generateContent", body,
                      {"Content-Type": "application/json", "x-goog-api-key": api_key}, timeout=timeout_seconds)
    return json.dumps(data, ensure_ascii=False), data.get("usageMetadata")


def academic_help_gemma_request(job: dict, *, timeout_seconds: float = 300.0) -> tuple[str, dict | None]:
    """Call the same official endpoint with the A4 candidate schema."""
    api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GEMMA_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY is required for the official Gemini generateContent endpoint")
    model = os.environ.get("GEMMA_MODEL", job["model"])
    endpoint = os.environ.get("GEMINI_BASE_URL", "https://generativelanguage.googleapis.com/v1beta")
    payload = job["payload"][0]
    body = {
        "contents": [{"parts": [{"text": build_academic_help_prompt(payload)}]}],
        "tools": [{"functionDeclarations": [academic_help_function_declaration()]}],
        "toolConfig": {"functionCallingConfig": {"mode": "ANY", "allowedFunctionNames": [ACADEMIC_HELP_FUNCTION_NAME]}},
    }
    data = _post_json(f"{endpoint.rstrip('/')}/models/{model}:generateContent", body,
                      {"Content-Type": "application/json", "x-goog-api-key": api_key}, timeout=timeout_seconds)
    return json.dumps(data, ensure_ascii=False), data.get("usageMetadata")


def gemini_request(job: dict, *, timeout_seconds: float = 300.0) -> tuple[str, dict | None]:
    return _gemini_generate_content_request(job, timeout_seconds=timeout_seconds)


def execute(args: argparse.Namespace, request_fn, response_parser=None) -> None:
    if args.max_retries < 0 or args.retry_delay < 0 or args.timeout_seconds <= 0:
        raise ValueError("retry options must be non-negative")
    jobs = load_jsonl(args.input)
    def log_request(number: int, total: int, rpm_limit: float, min_interval: float) -> None:
        rpm_text = f"{rpm_limit:g}"
        interval_text = f"{min_interval:g}"
        print(f"[{number}/{total}] request sent rate_limit={rpm_text} rpm next_request_in≈{interval_text}s", flush=True)

    def request_with_timeout(job: dict):
        return request_fn(job, timeout_seconds=args.timeout_seconds)

    records = run_jobs(jobs, args.output, request_with_timeout, resume=args.resume, max_retries=args.max_retries,
                       retry_delay=args.retry_delay, limit=args.limit, batch_size=args.batch_size,
                       rpm_limit=args.rpm_limit, request_logger=log_request, response_parser=response_parser)
    print(json.dumps({"jobs": len(jobs), "records": len(records), "output": str(args.output)}, indent=2))
