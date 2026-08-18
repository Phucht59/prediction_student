"""Provider-independent job execution with resume and finite transient retry."""

from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

from .constants import PROMPT_VERSION
from .parser import LabelParseError, parse_llm_response


class TransientAPIError(RuntimeError):
    """An error that may succeed on a finite retry."""

    def __init__(self, message: str, *, diagnostics: dict | None = None):
        super().__init__(message)
        self.diagnostics = diagnostics or {}


class RequestRateLimiter:
    """Space real request attempts evenly; skipped jobs never call ``wait``."""

    def __init__(self, rpm_limit: float = 12, *, clock=time.monotonic, sleep=time.sleep):
        if rpm_limit <= 0:
            raise ValueError("rpm_limit must be greater than zero")
        self.rpm_limit = float(rpm_limit)
        self.min_interval_seconds = 60.0 / self.rpm_limit
        self._clock = clock
        self._sleep = sleep
        self._last_request_at: float | None = None

    def wait_for_slot(self) -> None:
        now = self._clock()
        if self._last_request_at is not None:
            remaining = self.min_interval_seconds - (now - self._last_request_at)
            if remaining > 0:
                self._sleep(remaining)
        self._last_request_at = self._clock()


def load_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _completed_job_ids(records: list[dict]) -> set[str]:
    by_job: dict[str, list[dict]] = {}
    for record in records:
        by_job.setdefault(str(record.get("job_id")), []).append(record)
    return {job_id for job_id, rows in by_job.items() if rows and all(row.get("status") == "completed" for row in rows)}


def run_jobs(jobs: list[dict], output: Path, request_fn: Callable, *, resume=False, max_retries=3,
             retry_delay=2.0, limit=None, batch_size=10, rpm_limit=12,
             rate_limiter: RequestRateLimiter | None = None, request_logger: Callable | None = None,
             response_parser: Callable | None = None) -> list[dict]:
    output.parent.mkdir(parents=True, exist_ok=True)
    existing = load_jsonl(output) if resume and output.exists() else []
    completed = _completed_job_ids(existing)
    retained = [record for record in existing if str(record.get("job_id")) in completed]
    selected = [job for job in jobs if str(job["job_id"]) not in completed]
    if limit is not None:
        selected = selected[:limit]
    limiter = rate_limiter or RequestRateLimiter(rpm_limit)
    results = list(retained)
    for job_number, job in enumerate(selected, start=1):
        if len(job.get("case_ids", [])) > batch_size:
            raise ValueError(f"job {job['job_id']} exceeds --batch-size")
        expected_ids = [str(x) for x in job["case_ids"]]
        feasibility = {
            str(payload["case_id"]): payload["feasibility"]
            for payload in job["payload"]
            if "feasibility" in payload
        }
        last_error = None
        last_diagnostics: dict = {}
        last_raw_response = None
        last_attempt = 0
        for attempt in range(max_retries + 1):
            last_attempt = attempt + 1
            try:
                limiter.wait_for_slot()
                request_started = time.monotonic()
                try:
                    raw_response, token_usage = request_fn(job)
                finally:
                    request_elapsed = time.monotonic() - request_started
                    if request_logger is not None:
                        request_logger(job_number, len(selected), limiter.rpm_limit, limiter.min_interval_seconds)
                last_raw_response = raw_response
                if response_parser is None:
                    parsed = parse_llm_response(raw_response, expected_ids, feasibility,
                                                prompt_version=job.get("prompt_version") or PROMPT_VERSION)
                else:
                    parsed = response_parser(raw_response, expected_ids, feasibility)
                for case_id in expected_ids:
                    results.append({"job_id": job["job_id"], "case_id": case_id,
                                    "provider": job["provider"], "model": job["model"],
                                    "prompt_version": job["prompt_version"], "raw_response": raw_response,
                                    "parsed_labels": parsed[case_id], "status": "completed", "error": None,
                                    "timestamp": datetime.now(timezone.utc).isoformat(), "attempt": attempt + 1,
                                    **({"token_usage": token_usage} if token_usage is not None else {})})
                last_error = None
                break
            except TransientAPIError as exc:
                last_error = str(exc)
                last_diagnostics = dict(getattr(exc, "diagnostics", {}))
                last_diagnostics.setdefault("exception_class", exc.__class__.__name__)
                last_diagnostics.setdefault("elapsed_seconds", request_elapsed)
                if attempt < max_retries:
                    time.sleep(max(0.0, retry_delay) * (2 ** attempt))
            except (LabelParseError, ValueError, RuntimeError) as exc:
                last_error = str(exc)
                last_diagnostics = dict(getattr(exc, "diagnostics", {}))
                last_diagnostics.setdefault("exception_class", exc.__class__.__name__)
                last_diagnostics.setdefault("elapsed_seconds", request_elapsed)
                break
        if last_error is not None:
            for case_id in expected_ids:
                results.append({"job_id": job["job_id"], "case_id": case_id,
                                "provider": job["provider"], "model": job["model"],
                                "prompt_version": job["prompt_version"], "raw_response": last_raw_response,
                                "parsed_labels": None, "status": "failed", "error": last_error,
                                "timestamp": datetime.now(timezone.utc).isoformat(), "attempt": last_attempt,
                                **last_diagnostics})
        with output.open("w", encoding="utf-8", newline="\n") as handle:
            for record in results:
                handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
    if not selected:
        with output.open("w", encoding="utf-8", newline="\n") as handle:
            for record in results:
                handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
    return results
