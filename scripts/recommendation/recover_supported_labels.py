"""Offline recovery of supported Gemma/Gemini labels; never calls an API."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.recommendation.labeling.constants import ACTION_IDS, ABSTAIN_REASONS, PROMPT_VERSION  # noqa: E402
from src.recommendation.labeling.parser import LabelParseError, parse_llm_response  # noqa: E402
from src.recommendation.labeling.runtime import load_jsonl  # noqa: E402

SUPPORTED_ACTIONS = ("A1", "A2", "A3", "A5")
UNSUPPORTED_ACTION = "A4"


def _normalize_label(value):
    if type(value) is int and value in (0, 1, 2, 3):
        return str(value)
    if isinstance(value, str) and value in {"0", "1", "2", "3"}:
        return value
    if value == "ABSTAIN":
        return value
    raise LabelParseError(f"invalid label value: {value!r}")


def _validate_supported_item(item: dict, action_id: str, feasibility: str, *, allow_reason_mismatch: bool = False) -> tuple[str, str]:
    if not isinstance(item, dict) or "label" not in item:
        raise LabelParseError(f"missing label for {action_id}")
    label = _normalize_label(item["label"])
    raw_reason = item.get("reason")
    # The historical Gemini main artifact omits reason for some labels.  Keep
    # that metadata omission explicit as NONE; for a known infeasible action,
    # the required abstention reason is determined by the feasibility contract,
    # not invented as a relevance label.
    reason = raw_reason if raw_reason is not None else "NONE"
    if feasibility == "INFEASIBLE" and label == "ABSTAIN" and raw_reason is None:
        reason = "INFEASIBLE"
    if reason not in (*ABSTAIN_REASONS, "NONE"):
        raise LabelParseError(f"invalid reason for {action_id}: {reason!r}")
    if feasibility == "INFEASIBLE":
        if label != "ABSTAIN" or reason != "INFEASIBLE":
            raise LabelParseError(f"INFEASIBLE action is not ABSTAIN/INFEASIBLE: {action_id}")
    elif reason == "INFEASIBLE" and not allow_reason_mismatch:
        raise LabelParseError(f"INFEASIBLE reason is invalid for {feasibility}: {action_id}")
    if label == "ABSTAIN" and raw_reason is not None and reason not in ("INFEASIBLE", "INSUFFICIENT_INFORMATION"):
        raise LabelParseError(f"ABSTAIN requires an abstain reason: {action_id}")
    if label != "ABSTAIN" and reason != "NONE":
        raise LabelParseError(f"numeric label requires reason NONE: {action_id}")
    return label, reason


def _extract_single_gemma_case(raw_response: str, real_case_id: str) -> dict:
    try:
        data = json.loads(raw_response)
    except json.JSONDecodeError as exc:
        raise LabelParseError(f"malformed Gemma response JSON: {exc.msg}") from exc
    candidates = data.get("candidates") if isinstance(data, dict) else None
    parts = candidates[0].get("content", {}).get("parts", []) if isinstance(candidates, list) and candidates else []
    calls = [part.get("functionCall") for part in parts if isinstance(part, dict) and "functionCall" in part]
    if len(calls) != 1 or calls[0].get("name") != "submit_relevance_labels":
        raise LabelParseError("Gemma raw response must contain one submit_relevance_labels function call")
    args = calls[0].get("args")
    if not isinstance(args, dict) or set(args) != {"cases"} or not isinstance(args["cases"], list):
        raise LabelParseError("invalid Gemma functionCall args")
    if len(args["cases"]) != 1:
        raise LabelParseError("Gemma functionCall must contain one case")
    case = args["cases"][0]
    if not isinstance(case, dict) or set(case) != {"case_ref", "labels"} or case["case_ref"] != "C01":
        raise LabelParseError("Gemma case_ref must be exactly C01")
    labels = case["labels"]
    if not isinstance(labels, dict) or set(labels) != set(ACTION_IDS):
        raise LabelParseError("Gemma function call must contain exactly A1-A5")
    return {"case_id": real_case_id, "labels": labels}


def _feasibility_map(path: Path) -> dict[str, dict[str, str]]:
    frame = pd.read_parquet(path)
    compact = frame[["case_id", "action_id", "feasibility_status"]].copy()
    compact["case_id"] = compact["case_id"].astype(str)
    compact["action_id"] = compact["action_id"].astype(str)
    compact["feasibility_status"] = compact["feasibility_status"].astype(str)
    return {
        str(case_id): dict(zip(group["action_id"], group["feasibility_status"]))
        for case_id, group in compact.groupby("case_id", sort=False, observed=True)
    }


def _job_case_map(jobs_path: Path) -> dict[str, tuple[str, dict[str, str], str]]:
    result = {}
    for job in load_jsonl(jobs_path):
        case_ids = [str(case_id) for case_id in job["case_ids"]]
        if len(case_ids) != len(set(case_ids)):
            raise ValueError(f"duplicate case in job {job['job_id']}")
        for payload in job["payload"]:
            case_id = str(payload["case_id"])
            result[case_id] = (str(job["job_id"]), payload["feasibility"], str(job["prompt_version"]))
    return result


def _rows(case_id: str, labels: dict, feasibility: dict[str, str], lf_name: str, provider: str, model: str, prompt_version: str, *, allow_reason_mismatch: bool = False) -> list[dict]:
    result = []
    for action_id in SUPPORTED_ACTIONS:
        label, reason = _validate_supported_item(labels[action_id], action_id, feasibility[action_id], allow_reason_mismatch=allow_reason_mismatch)
        result.append({"case_id": case_id, "action_id": action_id, "lf_name": lf_name,
                       "label": label, "abstain": label == "ABSTAIN", "reason": reason,
                       "provider": provider, "model": model, "prompt_version": prompt_version,
                       "feasibility_status": feasibility[action_id]})
    return result


def recover_gemma(raw_path: Path, jobs_path: Path, feasibility_path: Path, output_path: Path, lf_name="LF_GEMMA") -> pd.DataFrame:
    jobs = _job_case_map(jobs_path)
    feasibility = _feasibility_map(feasibility_path)
    records = load_jsonl(raw_path)
    if len(records) != len(jobs):
        raise ValueError(f"Gemma raw records {len(records)} != expected jobs {len(jobs)}")
    rows = []
    seen = set()
    for record in records:
        case_id = str(record["case_id"])
        if case_id in seen or case_id not in jobs:
            raise ValueError(f"unexpected or duplicate Gemma case: {case_id}")
        seen.add(case_id)
        job_id, case_feasibility, prompt_version = jobs[case_id]
        if record["job_id"] != job_id:
            raise ValueError(f"job/case mismatch for {case_id}")
        parsed = _extract_single_gemma_case(record["raw_response"], case_id)
        rows.extend(_rows(case_id, parsed["labels"], case_feasibility, lf_name, record["provider"], record["model"], prompt_version))
    if seen != set(jobs):
        raise ValueError(f"missing Gemma cases: {len(set(jobs) - seen)}")
    frame = pd.DataFrame(rows).sort_values(["case_id", "action_id"])
    if len(frame) != 500 * len(SUPPORTED_ACTIONS) or set(frame["action_id"]) != set(SUPPORTED_ACTIONS):
        raise ValueError("Gemma supported-action row count is not 2000")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(output_path, index=False)
    return frame


def recover_gemini(raw_path: Path, jobs_path: Path, feasibility_path: Path, output_path: Path, lf_name="LF_GEMINI_MAIN") -> pd.DataFrame:
    jobs = load_jsonl(jobs_path)
    feasibility = _feasibility_map(feasibility_path)
    records = load_jsonl(raw_path)
    record_by_job = {}
    for record in records:
        record_by_job.setdefault(str(record["job_id"]), []).append(record)
    rows = []
    seen = set()
    for job in jobs:
        job_id = str(job["job_id"])
        job_records = record_by_job.get(job_id, [])
        raw_responses = [record.get("raw_response") for record in job_records if record.get("raw_response")]
        if not raw_responses:
            raise ValueError(f"missing Gemini raw response for {job_id}")
        expected_ids = [str(case_id) for case_id in job["case_ids"]]
        expected_feasibility = {case_id: feasibility[case_id] for case_id in expected_ids}
        # Parse the provider artifact first, then apply only the supported-action
        # contract below.  Historical Gemini rows contain an auxiliary
        # INFEASIBLE reason on some UNKNOWN A5 cases; this must not block
        # offline recovery of the label itself.
        parsed = parse_llm_response(raw_responses[0], expected_ids, prompt_version=job["prompt_version"])
        for case_id in expected_ids:
            if case_id in seen:
                raise ValueError(f"duplicate Gemini case: {case_id}")
            seen.add(case_id)
            record = next((item for item in job_records if str(item["case_id"]) == case_id), job_records[0])
            rows.extend(_rows(case_id, parsed[case_id]["labels"], feasibility[case_id], lf_name, record["provider"], record["model"], job["prompt_version"], allow_reason_mismatch=True))
    if len(seen) != 500:
        raise ValueError(f"Gemini main case count is {len(seen)}, expected 500")
    frame = pd.DataFrame(rows).sort_values(["case_id", "action_id"])
    if len(frame) != 500 * len(SUPPORTED_ACTIONS):
        raise ValueError("Gemini supported-action row count is not 2000")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(output_path, index=False)
    return frame


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gemma-raw", type=Path, default=ROOT / "artifacts/recommendation/labeling/raw/gemma_panel_a_single.jsonl")
    parser.add_argument("--gemma-jobs", type=Path, default=ROOT / "artifacts/recommendation/labeling/jobs/panel_a_gemma_single_jobs.jsonl")
    parser.add_argument("--gemini-raw", type=Path, default=ROOT / "artifacts/recommendation/labeling/raw/gemini_panel_a.jsonl")
    parser.add_argument("--gemini-jobs", type=Path, default=ROOT / "artifacts/recommendation/labeling/jobs/panel_a_gemini_jobs.jsonl")
    parser.add_argument("--feasibility", type=Path, default=ROOT / "artifacts/recommendation/feasibility/oulad_action_feasibility.parquet")
    parser.add_argument("--gemma-output", type=Path, default=ROOT / "artifacts/recommendation/labeling/normalized/gemma_supported_labels.parquet")
    parser.add_argument("--gemini-output", type=Path, default=ROOT / "artifacts/recommendation/labeling/normalized/gemini_supported_labels.parquet")
    args = parser.parse_args()
    gemma = recover_gemma(args.gemma_raw, args.gemma_jobs, args.feasibility, args.gemma_output)
    gemini = recover_gemini(args.gemini_raw, args.gemini_jobs, args.feasibility, args.gemini_output)
    print(json.dumps({"gemma_rows": len(gemma), "gemini_rows": len(gemini), "actions": list(SUPPORTED_ACTIONS)}, indent=2))


if __name__ == "__main__":
    main()
