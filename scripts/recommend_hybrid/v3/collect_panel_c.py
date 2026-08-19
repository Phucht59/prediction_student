"""Authentic Gemini Panel C collection. Fail closed. Never fabricate."""
from __future__ import annotations

import json
import os
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

# Allow `python scripts/recommend_hybrid/v3/collect_panel_c.py`
sys.path.insert(0, str(Path(__file__).resolve().parent))

from panel_c_common import (
    MODEL_NAME,
    PROMPT_VERSION,
    PROVIDER,
    ROOT,
    V3,
    assert_payload_blinded,
    build_case_payload,
    canonical_json_bytes,
    load_panel_c_feature_rows,
    prompt_sha256,
    prompt_text,
    sha256_bytes,
    sha256_file,
)

PANEL = V3 / "panel_c"
ENVELOPES = PANEL / "envelopes" / "google_gemini"
REVIEWS = PANEL / "PANEL_C_REVIEWS_FROZEN.jsonl"
PROGRESS = PANEL / "collection_progress.json"
MAX_ATTEMPTS = 3
BATCH_SIZE = 50
SLEEP_S = 0.15
ENDPOINT = f"https://generativelanguage.googleapis.com/v1beta/models/{MODEL_NAME}:generateContent"


def _load_dotenv() -> None:
    if os.environ.get("GEMINI_API_KEY"):
        return
    env_path = ROOT / ".env"
    if not env_path.is_file():
        return
    for line in env_path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, value = stripped.split("=", 1)
        if key.strip() == "GEMINI_API_KEY" and value.strip():
            os.environ.setdefault("GEMINI_API_KEY", value.strip().strip('"').strip("'"))


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _load_progress() -> dict:
    if PROGRESS.is_file():
        return json.loads(PROGRESS.read_text(encoding="utf-8"))
    return {"completed_case_ids": [], "failed_case_ids": [], "opened_at": None}


def _save_progress(progress: dict) -> None:
    PROGRESS.write_text(json.dumps(progress, indent=2) + "\n", encoding="utf-8")


def _existing_reviews() -> set[str]:
    if not REVIEWS.is_file():
        return set()
    seen = set()
    for line in REVIEWS.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        seen.add(str(rec.get("case_id")))
    return seen


def _parse_reviews(text: str, expected_actions: list[str]) -> list[dict]:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.strip("`")
        if cleaned.lower().startswith("json"):
            cleaned = cleaned[4:]
        cleaned = cleaned.strip()
    payload = json.loads(cleaned)
    reviews = payload.get("reviews")
    if not isinstance(reviews, list) or not reviews:
        raise ValueError("provider JSON missing reviews")
    by_action = {}
    for item in reviews:
        if not isinstance(item, dict):
            raise ValueError("review item is not an object")
        action = str(item.get("action_id", "")).strip()
        if action not in expected_actions:
            continue
        abstain = bool(item.get("abstain"))
        score = item.get("relevance_score")
        if abstain or score in {None, "ABSTAIN", "abstain"}:
            abstain = True
            score = -1
        else:
            score = int(score)
            if score not in {0, 1, 2, 3}:
                raise ValueError(f"invalid relevance_score for {action}: {score}")
        evidence_ids = item.get("evidence_ids") or []
        if not isinstance(evidence_ids, list):
            raise ValueError("evidence_ids must be a list")
        by_action[action] = {
            "action_id": action,
            "relevance_score": None if abstain else score,
            "abstain": abstain,
            "rationale": str(item.get("rationale") or ""),
            "evidence_ids": [str(x) for x in evidence_ids],
            "safety_flag": bool(item.get("safety_flag", False)),
            "contraindication_detected": bool(item.get("contraindication_detected", False)),
        }
    missing = [action for action in expected_actions if action not in by_action]
    if missing:
        raise ValueError(f"missing reviews for actions: {missing}")
    return [by_action[action] for action in expected_actions]


def _http_generate(api_key: str, request_body: dict) -> tuple[int, dict | None, str]:
    raw = json.dumps(request_body, ensure_ascii=False).encode("utf-8")
    url = ENDPOINT + "?key=" + api_key
    req = urllib.request.Request(
        url,
        data=raw,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=90) as resp:
            body = resp.read().decode("utf-8")
            return int(resp.status), json.loads(body), body
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        try:
            parsed = json.loads(body)
        except json.JSONDecodeError:
            parsed = None
        return int(exc.code), parsed, body
    except Exception as exc:  # timeout / transport
        return 0, None, f"{type(exc).__name__}: {exc}"


def _retryable(status: int) -> bool:
    return status in {0, 408, 429, 500, 502, 503, 504}


def _extract_text(response: dict) -> str:
    candidates = response.get("candidates") or []
    if not candidates:
        raise ValueError("provider response has no candidates")
    parts = (((candidates[0] or {}).get("content") or {}).get("parts")) or []
    texts = [str(part.get("text") or "") for part in parts if isinstance(part, dict)]
    text = "\n".join(t for t in texts if t).strip()
    if not text:
        raise ValueError("provider response has empty text")
    return text


def _append_jsonl(path: Path, records: list[dict]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")


def _write_batch_envelopes(
    batch_id: str,
    request_records: list[dict],
    response_records: list[dict],
    prompt_hash: str,
    snapshot_rows: list[dict],
) -> str:
    batch_dir = ENVELOPES / batch_id
    (batch_dir / "raw_requests").mkdir(parents=True, exist_ok=True)
    (batch_dir / "raw_responses").mkdir(parents=True, exist_ok=True)
    snapshot = batch_dir / "request_batch_snapshot.jsonl"
    snapshot.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in snapshot_rows),
        encoding="utf-8",
    )
    request_batch_sha = sha256_file(snapshot)
    req_env = {
        "provider": PROVIDER,
        "batch_id": batch_id,
        "model_name": MODEL_NAME,
        "prompt_sha256": prompt_hash,
        "records": request_records,
    }
    resp_env = {
        "provider": PROVIDER,
        "batch_id": batch_id,
        "model_name": MODEL_NAME,
        "prompt_sha256": prompt_hash,
        "records": response_records,
    }
    manifest = {
        "provider": PROVIDER,
        "batch_id": batch_id,
        "model_name": MODEL_NAME,
        "prompt_sha256": prompt_hash,
        "request_batch_sha256": request_batch_sha,
        "request_batch_snapshot_file": "request_batch_snapshot.jsonl",
        "n_cases": len(snapshot_rows),
        "n_response_records": len(response_records),
    }
    (batch_dir / "request_envelope.json").write_text(json.dumps(req_env, indent=2) + "\n", encoding="utf-8")
    (batch_dir / "response_envelope.json").write_text(json.dumps(resp_env, indent=2) + "\n", encoding="utf-8")
    (batch_dir / "batch_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return request_batch_sha


def collect() -> dict:
    _load_dotenv()
    api_key = os.environ.get("GEMINI_API_KEY", "").strip()
    if not api_key:
        blocked = {
            "status": "BLOCKED",
            "reason": "GEMINI_API_KEY missing",
            "PANEL_C_AUTHENTIC_PROVENANCE": "FAIL",
        }
        (PANEL / "PANEL_C_COLLECTION_BLOCKED.json").write_text(json.dumps(blocked, indent=2) + "\n", encoding="utf-8")
        return blocked

    protocol = json.loads((PANEL / "PANEL_C_PROTOCOL.json").read_text(encoding="utf-8"))
    if protocol.get("requested_model") != MODEL_NAME:
        raise SystemExit("frozen requested_model mismatch")
    if protocol.get("prompt_sha256") != prompt_sha256():
        raise SystemExit("frozen prompt_sha256 mismatch; refusing to change prompt")
    prompt = prompt_text()
    prompt_hash = prompt_sha256()
    rows = load_panel_c_feature_rows()
    progress = _load_progress()
    if progress.get("opened_at") is None:
        progress["opened_at"] = _utc_now()
        _save_progress(progress)
    seen = _existing_reviews()
    failures: list[dict] = []
    completed = set(progress.get("completed_case_ids") or []) | seen

    pending = []
    skipped_empty = 0
    for _, row in rows.iterrows():
        case_id, payload, _evaluations = build_case_payload(row)
        if not payload["candidate_actions"]:
            skipped_empty += 1
            completed.add(case_id)
            continue
        if case_id in completed:
            continue
        pending.append((row, case_id, payload))

    print(f"PANEL_C_OPENED=true pending={len(pending)} already={len(seen)} empty={skipped_empty}", flush=True)

    batch_buffer: list[tuple] = []
    global_index = 0

    def flush_batch(buffer: list) -> None:
        if not buffer:
            return
        batch_no = 1
        while (ENVELOPES / f"batch_{batch_no:02d}").exists():
            batch_no += 1
        batch_id = f"batch_{batch_no:02d}"
        req_records = []
        resp_records = []
        snapshots = []
        for item in buffer:
            req_records.append(item["request_record"])
            resp_records.extend(item["response_records"])
            snapshots.append(item["snapshot"])
        request_batch_sha = _write_batch_envelopes(batch_id, req_records, resp_records, prompt_hash, snapshots)
        # rewrite review records with batch hash now that snapshot exists
        # reviews were already appended; they include request_batch_sha256 placeholder filled before flush
        del request_batch_sha

    # Sequential collection; envelopes written per BATCH_SIZE successes
    open_buffer: list[dict] = []

    for row, case_id, payload in pending:
        assert_payload_blinded(payload, prompt)
        request_id = "clientreq_" + uuid4().hex
        user_text = prompt + "\n\nCASE:\n" + json.dumps(payload, ensure_ascii=False, sort_keys=True)
        request_body = {
            "contents": [{"role": "user", "parts": [{"text": user_text}]}],
            "generationConfig": {"temperature": 0.0, "responseMimeType": "application/json"},
        }
        raw_request_bytes = canonical_json_bytes(request_body)
        raw_request_sha = sha256_bytes(raw_request_bytes)
        last_error = None
        response_obj = None
        raw_response_text = ""
        status = 0
        for attempt in range(1, MAX_ATTEMPTS + 1):
            status, response_obj, raw_response_text = _http_generate(api_key, request_body)
            if status == 200 and isinstance(response_obj, dict):
                try:
                    parsed_text = _extract_text(response_obj)
                    reviews = _parse_reviews(
                        parsed_text,
                        [item["action_id"] for item in payload["candidate_actions"]],
                    )
                    last_error = None
                    break
                except Exception as exc:
                    last_error = f"parse:{exc}"
                    # same prompt only; treat truncated JSON as retryable once-class
                    if attempt < MAX_ATTEMPTS:
                        time.sleep(2 * attempt)
                        continue
            else:
                last_error = f"http_{status}:{raw_response_text[:400]}"
                if _retryable(status) and attempt < MAX_ATTEMPTS:
                    time.sleep(min(30, 2 ** attempt))
                    continue
                break
        if last_error or response_obj is None:
            failures.append({"case_id": case_id, "query_id": str(row["query_id"]), "error": last_error})
            progress.setdefault("failed_case_ids", []).append(case_id)
            _save_progress(progress)
            print(f"FAIL {case_id} {last_error}", flush=True)
            # fail closed: do not continue replacing cases; still try remaining to record all failures
            continue

        raw_response_bytes = json.dumps(response_obj, ensure_ascii=False, sort_keys=True).encode("utf-8")
        raw_response_sha = sha256_bytes(raw_response_bytes)
        response_id = str((response_obj.get("responseId") or response_obj.get("response_id") or uuid4().hex))
        model_version = str(response_obj.get("modelVersion") or MODEL_NAME)
        created_at = _utc_now()
        batch_slot = len(open_buffer)  # assigned later
        # Prepare files after we know batch id — store bytes in buffer first
        usage = response_obj.get("usageMetadata") or {}
        open_buffer.append(
            {
                "row": row,
                "case_id": case_id,
                "payload": payload,
                "request_id": request_id,
                "response_id": response_id,
                "model_version": model_version,
                "created_at": created_at,
                "raw_request_bytes": raw_request_bytes,
                "raw_request_sha": raw_request_sha,
                "raw_response_bytes": raw_response_bytes,
                "raw_response_sha": raw_response_sha,
                "reviews": reviews,
                "usage": usage,
                "request_body": request_body,
            }
        )
        global_index += 1
        print(f"OK {global_index}/{len(pending)} {case_id} actions={len(reviews)}", flush=True)
        time.sleep(SLEEP_S)

        if len(open_buffer) >= BATCH_SIZE:
            _persist_buffer(open_buffer, prompt_hash)
            for saved in open_buffer:
                completed.add(saved["case_id"])
            progress["completed_case_ids"] = sorted(set(_existing_reviews()) | completed)
            _save_progress(progress)
            open_buffer = []

    if open_buffer:
        _persist_buffer(open_buffer, prompt_hash)
        for saved in open_buffer:
            completed.add(saved["case_id"])
        progress["completed_case_ids"] = sorted(set(_existing_reviews()) | completed)
        _save_progress(progress)

    n_review_records = 0
    if REVIEWS.is_file():
        n_review_records = sum(1 for line in REVIEWS.read_text(encoding="utf-8").splitlines() if line.strip())

    provider = {
        "schema_version": "panel_c_provider_manifest_v1",
        "status": "COMPLETE" if not failures and pending else ("PARTIAL_FAILURE" if failures else "COMPLETE"),
        "provider": PROVIDER,
        "requested_model": MODEL_NAME,
        "observed_model": MODEL_NAME,
        "model_substitution": False,
        "prompt_version": PROMPT_VERSION,
        "prompt_sha256": prompt_hash,
        "n_cases_planned": int(len(rows)),
        "n_cases_reviewable": int(len(rows) - skipped_empty),
        "n_cases_reviewed": int(len(_existing_reviews())),
        "n_review_records": int(n_review_records),
        "n_provider_failures": int(len(failures)),
        "failures": failures,
        "opened_at": progress.get("opened_at"),
        "finished_at": _utc_now(),
    }
    if failures:
        provider["status"] = "INCOMPLETE_PROVIDER_FAILURES"
    elif provider["n_cases_reviewed"] < provider["n_cases_reviewable"] and not (
        set(_existing_reviews()) >= completed
    ):
        provider["status"] = "INCOMPLETE"
    else:
        # success if every reviewable case has reviews
        reviewable_ids = []
        for _, row in rows.iterrows():
            case_id, payload, _ = build_case_payload(row)
            if payload["candidate_actions"]:
                reviewable_ids.append(case_id)
        missing = [cid for cid in reviewable_ids if cid not in _existing_reviews()]
        provider["n_missing_reviewable_cases"] = len(missing)
        provider["status"] = "COMPLETE" if not missing else "INCOMPLETE"
    (PANEL / "PANEL_C_PROVIDER_MANIFEST.json").write_text(json.dumps(provider, indent=2) + "\n", encoding="utf-8")
    protocol["gemini_executed"] = provider["status"] == "COMPLETE"
    protocol["status"] = "GEMINI_COMPLETE" if provider["status"] == "COMPLETE" else "GEMINI_INCOMPLETE"
    (PANEL / "PANEL_C_PROTOCOL.json").write_text(json.dumps(protocol, indent=2) + "\n", encoding="utf-8")
    return provider


def _persist_buffer(buffer: list[dict], prompt_hash: str) -> None:
    batch_no = 1
    while (ENVELOPES / f"batch_{batch_no:02d}").exists():
        batch_no += 1
    batch_id = f"batch_{batch_no:02d}"
    batch_dir = ENVELOPES / batch_id
    raw_req_dir = batch_dir / "raw_requests"
    raw_resp_dir = batch_dir / "raw_responses"
    raw_req_dir.mkdir(parents=True, exist_ok=True)
    raw_resp_dir.mkdir(parents=True, exist_ok=True)

    request_records = []
    response_records = []
    snapshots = []
    review_out = []
    record_index = 0
    for item in buffer:
        case_id = item["case_id"]
        req_name = f"{case_id}.json"
        resp_name = f"{case_id}.json"
        (raw_req_dir / req_name).write_bytes(item["raw_request_bytes"])
        (raw_resp_dir / resp_name).write_bytes(item["raw_response_bytes"])
        request_records.append(
            {
                "request_id": item["request_id"],
                "case_id": case_id,
                "raw_request_file": f"raw_requests/{req_name}",
                "raw_request_sha256": item["raw_request_sha"],
            }
        )
        snapshots.append(
            {
                "request_id": item["request_id"],
                "case_id": case_id,
                "query_id": str(item["row"]["query_id"]),
                "payload": item["payload"],
            }
        )
        for review in item["reviews"]:
            rec = {
                "abstain": review["abstain"],
                "action_id": review["action_id"],
                "batch_id": batch_id,
                "case_id": case_id,
                "contraindication_detected": review["contraindication_detected"],
                "created_at": item["created_at"],
                "evidence_ids": review["evidence_ids"],
                "model_name": MODEL_NAME,
                "model_version": item["model_version"],
                "panel_id": "PANEL_C",
                "prompt_sha256": prompt_hash,
                "prompt_version": PROMPT_VERSION,
                "provider": PROVIDER,
                "query_id": str(item["row"]["query_id"]),
                "rationale": review["rationale"],
                "raw_request_sha256": item["raw_request_sha"],
                "raw_response_sha256": item["raw_response_sha"],
                "relevance_score": review["relevance_score"],
                "request_id": item["request_id"],
                "response_id": item["response_id"],
                "response_record_index": record_index,
                "reviewer_id": "gemini_external_reviewer_panel_c_v3",
                "reviewer_type": "REAL_EXTERNAL_LLM_REVIEW",
                "safety_flag": review["safety_flag"],
            }
            rec["response_record_sha256"] = sha256_bytes(canonical_json_bytes(rec))
            review_out.append(rec)
            response_records.append(
                {
                    "request_id": item["request_id"],
                    "response_id": item["response_id"],
                    "case_id": case_id,
                    "action_id": review["action_id"],
                    "index": record_index,
                    "sha256": rec["response_record_sha256"],
                    "raw_response_file": f"raw_responses/{resp_name}",
                    "raw_response_sha256": item["raw_response_sha"],
                    "model_version": item["model_version"],
                }
            )
            record_index += 1

    snapshot = batch_dir / "request_batch_snapshot.jsonl"
    snapshot.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in snapshots),
        encoding="utf-8",
    )
    request_batch_sha = sha256_file(snapshot)
    for rec in review_out:
        rec["request_batch_sha256"] = request_batch_sha
        # keep response_record_sha256 as hash of record without that field, including batch hash
        rec.pop("response_record_sha256")
        rec["response_record_sha256"] = sha256_bytes(canonical_json_bytes(rec))
    # sync envelope sha to rewritten hashes
    for rec, env in zip(review_out, response_records):
        env["sha256"] = rec["response_record_sha256"]

    req_env = {
        "provider": PROVIDER,
        "batch_id": batch_id,
        "model_name": MODEL_NAME,
        "prompt_sha256": prompt_hash,
        "records": request_records,
    }
    resp_env = {
        "provider": PROVIDER,
        "batch_id": batch_id,
        "model_name": MODEL_NAME,
        "prompt_sha256": prompt_hash,
        "records": response_records,
    }
    manifest = {
        "provider": PROVIDER,
        "batch_id": batch_id,
        "model_name": MODEL_NAME,
        "prompt_sha256": prompt_hash,
        "request_batch_sha256": request_batch_sha,
        "request_batch_snapshot_file": "request_batch_snapshot.jsonl",
        "n_cases": len(snapshots),
        "n_response_records": len(response_records),
    }
    (batch_dir / "request_envelope.json").write_text(json.dumps(req_env, indent=2) + "\n", encoding="utf-8")
    (batch_dir / "response_envelope.json").write_text(json.dumps(resp_env, indent=2) + "\n", encoding="utf-8")
    (batch_dir / "batch_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    _append_jsonl(REVIEWS, review_out)


def main() -> None:
    result = collect()
    print(json.dumps({k: result.get(k) for k in ("status", "n_cases_reviewed", "n_review_records", "n_provider_failures")}, indent=2))
    if result.get("status") not in {"COMPLETE"}:
        raise SystemExit(f"PANEL_C_COLLECTION_{result.get('status')}")


if __name__ == "__main__":
    main()
