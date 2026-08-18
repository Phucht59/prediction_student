from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from scripts.recommendation.build_labeling_jobs import ROOT, choose_pilot
from scripts.recommendation.build_weak_label_table import build as build_normalized
from scripts.recommendation import _run_labeling as runner
from scripts.recommendation._run_labeling import common_parser
from src.recommendation.labeling.constants import ACTION_IDS, PROMPT_VERSION
from src.recommendation.labeling.parser import LabelParseError, parse_llm_response
from src.recommendation.labeling.runtime import RequestRateLimiter, run_jobs


JOBS = ROOT / "artifacts/recommendation/labeling/jobs"
SINGLE_GEMMA_JOBS = JOBS / "pilot_gemma_single_jobs.jsonl"


def _read(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def test_jobs_cover_panel_a_only_and_are_reproducible():
    panel_a = set(pd.read_parquet(ROOT / "artifacts/recommendation/panels/panel_a.parquet")["case_id"].astype(str))
    panel_b = set(pd.read_parquet(ROOT / "artifacts/recommendation/panels/panel_b.parquet")["case_id"].astype(str))
    gemma = _read(JOBS / "panel_a_gemma_jobs.jsonl")
    gemini = _read(JOBS / "panel_a_gemini_jobs.jsonl")
    assert set(sum((job["case_ids"] for job in gemma), [])) == panel_a
    assert not set(sum((job["case_ids"] for job in gemma), [])) & panel_b
    assert [job["case_ids"] for job in gemma] == [job["case_ids"] for job in gemini]
    assert all(job["prompt_version"] == PROMPT_VERSION for job in gemma + gemini)
    assert all("GEMINI_API_KEY" not in json.dumps(job) for job in gemma + gemini)
    pilot = _read(JOBS / "pilot_gemini_jobs.jsonl")
    pilot_ids = set(sum((job["case_ids"] for job in pilot), []))
    panel = pd.read_parquet(ROOT / "artifacts/recommendation/panels/panel_a.parquet")
    pilot_rows = panel[panel["case_id"].astype(str).isin(pilot_ids)]
    assert len(pilot_ids) == 30
    assert set(pilot_rows["stage"]) == {"20pct", "35pct", "50pct", "75pct"}
    assert set(pilot_rows["outer_fold"]) == {0, 1, 2}
    assert set(pilot_rows["sampling_risk_band"]) == {"Low", "Borderline", "High"}


def test_strict_parser_and_feasibility_contract():
    case_id = "c1"
    labels = {}
    for action_id in ACTION_IDS:
        if action_id == "A1":
            labels[action_id] = {"label": "ABSTAIN", "reason": "INFEASIBLE"}
        else:
            labels[action_id] = {"label": 2, "confidence": "MEDIUM"}
    raw = json.dumps({"results": [{"case_id": case_id, "prompt_version": PROMPT_VERSION, "labels": labels}]})
    parsed = parse_llm_response(raw, [case_id], {case_id: {"A1": "INFEASIBLE", "A2": "FEASIBLE", "A3": "FEASIBLE", "A4": "UNKNOWN", "A5": "UNKNOWN"}})
    assert parsed[case_id]["labels"]["A1"]["label"] == "ABSTAIN"
    bad = raw.replace('"ABSTAIN"', "0")
    with pytest.raises(LabelParseError):
        parse_llm_response(bad, [case_id], {case_id: {"A1": "INFEASIBLE"}})
    with pytest.raises(LabelParseError):
        parse_llm_response("not-json", [case_id])


def test_parser_normalizes_numeric_strings_without_relaxing_values():
    labels = {action_id: {"label": str(index), "confidence": "LOW"}
              for index, action_id in enumerate(ACTION_IDS[:4])}
    labels["A5"] = {"label": "ABSTAIN", "confidence": "LOW", "reason": "INSUFFICIENT_INFORMATION"}
    raw = json.dumps({"case_id": "c1", "prompt_version": PROMPT_VERSION, "labels": labels})
    parsed = parse_llm_response(raw, ["c1"], {"c1": {action_id: "FEASIBLE" for action_id in ACTION_IDS}})
    assert [parsed["c1"]["labels"][action_id]["label"] for action_id in ACTION_IDS] == [0, 1, 2, 3, "ABSTAIN"]
    for invalid in (True, 1.0, "01", "4", None):
        invalid_labels = {action_id: {"label": 1} for action_id in ACTION_IDS}
        invalid_labels["A1"]["label"] = invalid
        invalid_raw = json.dumps({"case_id": "c1", "prompt_version": PROMPT_VERSION, "labels": invalid_labels})
        with pytest.raises(LabelParseError):
            parse_llm_response(invalid_raw, ["c1"])


def test_runtime_resume_and_normalized_grain(tmp_path):
    job = _read(SINGLE_GEMMA_JOBS)[0]
    output = tmp_path / "raw.jsonl"
    calls = []

    def fake_request(current):
        calls.append(current["job_id"])
        results = []
        for payload in current["payload"]:
            labels = {}
            for action_id, status in payload["feasibility"].items():
                labels[action_id] = ({"label": "ABSTAIN", "reason": "INFEASIBLE"}
                                     if status == "INFEASIBLE" else {"label": 1})
            results.append({"case_id": payload["case_id"], "prompt_version": PROMPT_VERSION, "labels": labels})
        return json.dumps({"results": results}), {"total_tokens": 1}

    run_jobs([job], output, fake_request, max_retries=0, retry_delay=0, batch_size=10)
    run_jobs([job], output, fake_request, resume=True, max_retries=0, retry_delay=0, batch_size=10)
    assert calls == [job["job_id"]]
    raw_records = _read(output)
    assert len(raw_records) == len(job["case_ids"])
    assert all(record["status"] == "completed" for record in raw_records)
    normalized = build_normalized(
        output, tmp_path / "labels.parquet", "LF_GEMMA",
        ROOT / "artifacts/recommendation/states/oulad_student_states.parquet",
        ROOT / "artifacts/recommendation/feasibility/oulad_action_feasibility.parquet",
    )
    assert len(normalized) == len(job["case_ids"]) * 5
    assert not normalized.duplicated(["case_id", "action_id", "lf_name"]).any()
    assert all(normalized.loc[normalized["feasibility_status"] == "INFEASIBLE", "label"] == "ABSTAIN")


def test_prompt_and_pilot_are_deterministic():
    panel = pd.read_parquet(ROOT / "artifacts/recommendation/panels/panel_a.parquet")
    left = choose_pilot(panel, 30, 2026)
    right = choose_pilot(panel, 30, 2026)
    assert left["case_id"].tolist() == right["case_id"].tolist()
    gemma = _read(JOBS / "panel_a_gemma_jobs.jsonl")
    gemini = _read(JOBS / "panel_a_gemini_jobs.jsonl")
    assert [job["prompt"] for job in gemma] == [job["prompt"] for job in gemini]


def test_rate_limit_default_and_interval():
    parser = common_parser("test")
    args = parser.parse_args(["--input", "in.jsonl", "--output", "out.jsonl"])
    assert args.rpm_limit == 12
    assert RequestRateLimiter(args.rpm_limit).min_interval_seconds == 5


def test_gemma_default_rate_limit_is_27_rpm_without_changing_shared_gemini_default():
    gemma_parser = common_parser("gemma", default_rpm=27.0)
    gemma_args = gemma_parser.parse_args(["--input", "in.jsonl", "--output", "out.jsonl"])
    assert gemma_args.rpm_limit == 27
    assert RequestRateLimiter(gemma_args.rpm_limit).min_interval_seconds == pytest.approx(60 / 27)


def test_rate_limit_custom_rpm_and_spacing():
    now = [0.0]
    sleeps = []

    def clock():
        return now[0]

    def sleep(seconds):
        sleeps.append(seconds)
        now[0] += seconds

    limiter = RequestRateLimiter(30, clock=clock, sleep=sleep)
    limiter.wait_for_slot()
    now[0] += 1
    limiter.wait_for_slot()
    assert limiter.min_interval_seconds == 2
    assert sleeps == [1]


def test_rate_limit_rejects_non_positive_rpm():
    parser = common_parser("test")
    with pytest.raises(SystemExit):
        parser.parse_args(["--input", "in.jsonl", "--output", "out.jsonl", "--rpm-limit", "0"])
    with pytest.raises(SystemExit):
        parser.parse_args(["--input", "in.jsonl", "--output", "out.jsonl", "--rpm-limit", "-1"])
    with pytest.raises(ValueError):
        RequestRateLimiter(0)


def test_timeout_option_and_official_gemma_endpoint(monkeypatch):
    parser = common_parser("test")
    args = parser.parse_args(["--input", "in.jsonl", "--output", "out.jsonl"])
    assert args.timeout_seconds == 300
    with pytest.raises(SystemExit):
        parser.parse_args(["--input", "in.jsonl", "--output", "out.jsonl", "--timeout-seconds", "0"])
    captured = {}

    def fake_post(url, body, headers, timeout):
        captured.update({"url": url, "body": body, "timeout": timeout})
        return {"candidates": [{"content": {"parts": [{"text": "{}"}]}}]}

    monkeypatch.setenv("GEMINI_API_KEY", "test-only-key")
    monkeypatch.setattr(runner, "_post_json", fake_post)
    job = _read(SINGLE_GEMMA_JOBS)[0]
    test_job = {"model": "gemma-4-31b-it", "prompt": job["prompt"], "case_ids": job["case_ids"]}
    raw = runner.gemma_request(test_job, timeout_seconds=321)[0]
    assert captured["url"].startswith("https://generativelanguage.googleapis.com/v1beta/models/gemma-4-31b-it:generateContent")
    assert captured["timeout"] == 321
    assert len(captured["body"]["tools"]) == 1
    declaration = captured["body"]["tools"][0]["functionDeclarations"]
    assert [function["name"] for function in declaration] == [runner.GEMMA_FUNCTION_NAME]
    assert captured["body"]["toolConfig"]["functionCallingConfig"] == {
        "mode": "ANY", "allowedFunctionNames": [runner.GEMMA_FUNCTION_NAME]
    }
    assert "generationConfig" not in captured["body"]
    assert raw.startswith('{"candidates"')


def test_gemma_function_call_args_are_parsed_and_prose_is_rejected():
    job = next(job for job in _read(SINGLE_GEMMA_JOBS) if job["payload"][0]["feasibility"]["A1"] == "INFEASIBLE")
    cases = []
    feasibility = {}
    for payload in job["payload"]:
        labels = {}
        feasibility[payload["case_id"]] = payload["feasibility"]
        for action_id, status in payload["feasibility"].items():
            labels[action_id] = {"label": "ABSTAIN", "reason": "INFEASIBLE"} if status == "INFEASIBLE" else {"label": "1", "reason": "NONE"}
        cases.append({"case_ref": f"C{len(cases) + 1:02d}", "labels": labels})
    response = {"candidates": [{"content": {"parts": [{"functionCall": {
        "name": runner.GEMMA_FUNCTION_NAME, "args": {"cases": cases}
    }}]}}]}
    parsed = runner.parse_gemma_function_call(json.dumps(response), job["case_ids"], feasibility)
    assert set(parsed) == set(job["case_ids"])
    assert parsed[job["case_ids"][0]]["labels"]["A2"]["label"] == 1
    alias_prompt = runner.build_gemma_prompt(job)
    assert "C01" in alias_prompt
    assert "C02" not in alias_prompt
    assert not any(case_id in alias_prompt for case_id in job["case_ids"])
    assert "UNKNOWN does not mean INFEASIBLE." in alias_prompt
    corrupted = json.loads(json.dumps(response))
    corrupted["candidates"][0]["content"]["parts"][0]["functionCall"]["args"]["cases"][0]["case_ref"] = "C99"
    with pytest.raises(LabelParseError):
        runner.parse_gemma_function_call(json.dumps(corrupted), job["case_ids"], feasibility)
    duplicated = json.loads(json.dumps(response))
    duplicated_case = json.loads(json.dumps(cases[0]))
    duplicated_case["case_ref"] = "C01"
    duplicated["candidates"][0]["content"]["parts"][0]["functionCall"]["args"]["cases"].append(duplicated_case)
    with pytest.raises(LabelParseError):
        runner.parse_gemma_function_call(json.dumps(duplicated), job["case_ids"], feasibility)
    missing = json.loads(json.dumps(response))
    missing["candidates"][0]["content"]["parts"][0]["functionCall"]["args"]["cases"] = missing["candidates"][0]["content"]["parts"][0]["functionCall"]["args"]["cases"][:-1]
    with pytest.raises(LabelParseError):
        runner.parse_gemma_function_call(json.dumps(missing), job["case_ids"], feasibility)
    unknown_bad = json.loads(json.dumps(response))
    unknown_bad["candidates"][0]["content"]["parts"][0]["functionCall"]["args"]["cases"][0]["labels"]["A4"] = {"label": "ABSTAIN", "reason": "INFEASIBLE"}
    with pytest.raises(LabelParseError):
        runner.parse_gemma_function_call(json.dumps(unknown_bad), job["case_ids"], feasibility)
    unknown_ok = json.loads(json.dumps(response))
    unknown_ok["candidates"][0]["content"]["parts"][0]["functionCall"]["args"]["cases"][0]["labels"]["A4"] = {"label": "ABSTAIN", "reason": "INSUFFICIENT_INFORMATION"}
    assert runner.parse_gemma_function_call(json.dumps(unknown_ok), job["case_ids"], feasibility)
    infeasible_ok = json.loads(json.dumps(response))
    infeasible_ok["candidates"][0]["content"]["parts"][0]["functionCall"]["args"]["cases"][0]["labels"]["A1"] = {"label": "ABSTAIN", "reason": "INFEASIBLE"}
    assert runner.parse_gemma_function_call(json.dumps(infeasible_ok), job["case_ids"], feasibility)
    with pytest.raises(LabelParseError):
        runner.parse_gemma_function_call(json.dumps({"candidates": [{"content": {"parts": [{"text": "A1=3"}]}}]}), job["case_ids"], feasibility)


def test_gemma_function_schema_required_keys_exist_in_same_properties():
    schema = runner._gemma_function_declaration()

    def walk(node):
        if not isinstance(node, dict):
            return
        if "required" in node:
            assert set(node["required"]).issubset(set(node.get("properties", {})))
        for child in node.get("properties", {}).values():
            walk(child)
        items = node.get("items")
        if items is not None:
            walk(items)

    walk(schema["parameters"])
    case_item = schema["parameters"]["properties"]["cases"]["items"]
    assert set(case_item["properties"]) == {"case_ref", "labels"}
    assert "case_id" not in json.dumps(schema)


def test_api_failure_diagnostics_are_persisted(tmp_path):
    job = _read(SINGLE_GEMMA_JOBS)[0]
    output = tmp_path / "raw.jsonl"

    def failing_request(_):
        raise runner.TransientAPIRequestError("HTTP 503", status_code=503,
                                               response_body='{"error":"busy"}', elapsed_seconds=1.25)

    run_jobs([job], output, failing_request, max_retries=0, retry_delay=0, rpm_limit=12)
    records = _read(output)
    assert records[0]["status"] == "failed"
    assert records[0]["exception_class"] == "TransientAPIRequestError"
    assert records[0]["http_status"] == 503
    assert records[0]["response_body"] == '{"error":"busy"}'
    assert records[0]["elapsed_seconds"] == 1.25
    assert records[0]["attempt"] == 1


def test_resume_skips_rate_limiter_for_completed_jobs(tmp_path):
    job = _read(JOBS / "pilot_gemma_jobs.jsonl")[0]
    output = tmp_path / "raw.jsonl"

    def fake_request(current):
        results = []
        for payload in current["payload"]:
            labels = {action_id: ({"label": "ABSTAIN", "reason": "INFEASIBLE"}
                                  if status == "INFEASIBLE" else {"label": 1})
                      for action_id, status in payload["feasibility"].items()}
            results.append({"case_id": payload["case_id"], "prompt_version": PROMPT_VERSION, "labels": labels})
        return json.dumps({"results": results}), None

    run_jobs([job], output, fake_request, max_retries=0, retry_delay=0, rpm_limit=12)
    sleeps = []
    limiter = RequestRateLimiter(12, clock=lambda: 0.0, sleep=sleeps.append)
    run_jobs([job], output, lambda _: pytest.fail("resumed job must be skipped"), resume=True,
             max_retries=0, retry_delay=0, rate_limiter=limiter)
    assert sleeps == []


def test_old_failed_gemma_job_is_resumable(tmp_path):
    job = _read(JOBS / "pilot_gemma_jobs.jsonl")[0]
    old_path = ROOT / "artifacts/recommendation/labeling/raw/gemma_pilot.jsonl"
    old_records = [record for record in _read(old_path) if record["job_id"] == job["job_id"]]
    assert old_records and all(record["status"] == "failed" for record in old_records)
    output = tmp_path / "old_failed.jsonl"
    output.write_text("\n".join(json.dumps(record) for record in old_records) + "\n", encoding="utf-8")
    calls = []

    def fake_request(current):
        calls.append(current["job_id"])
        results = []
        for payload in current["payload"]:
            labels = {action_id: ({"label": "ABSTAIN", "reason": "INFEASIBLE"}
                                  if status == "INFEASIBLE" else {"label": 1})
                      for action_id, status in payload["feasibility"].items()}
            results.append({"case_id": payload["case_id"], "prompt_version": PROMPT_VERSION, "labels": labels})
        return json.dumps({"results": results}), None

    run_jobs([job], output, fake_request, resume=True, max_retries=0, retry_delay=0)
    assert calls == [job["job_id"]]
