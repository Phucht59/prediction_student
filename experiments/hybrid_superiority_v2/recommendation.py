"""Gemini weak-label quota. Prediction HPO never calls this."""
from __future__ import annotations

import hashlib
from datetime import datetime, timezone
from typing import Any

SAFE_CAP = 480
HARD_CAP = 500
RESERVE = 20
ALIASES = {
    "gemini_3_5_flash_lite": "Gemini 3.5 Flash Lite",
    "gemini_3_1_flash_lite": "Gemini 3.1 Flash Lite",
}


def quota_day_utc(now: datetime | None = None) -> str:
    stamp = now or datetime.now(timezone.utc)
    return stamp.strftime("%Y-%m-%d")


def idempotency_key(case_hash: str, model_id: str, prompt_hash: str, schema_version: str) -> str:
    raw = "|".join([case_hash, model_id, prompt_hash, schema_version])
    return hashlib.sha256(raw.encode()).hexdigest()


def remaining(attempted: int, successful: int, *, safe_cap: int = SAFE_CAP, hard_cap: int = HARD_CAP) -> int:
    used = max(attempted, successful)
    return max(0, min(safe_cap, hard_cap) - used)


def allow_request(attempted: int, successful: int) -> bool:
    return remaining(attempted, successful) > 0


def validate_ranking_payload(payload: dict[str, Any]) -> tuple[bool, list[str]]:
    errors = []
    if not isinstance(payload, dict):
        return False, ["not_object"]
    ranking = payload.get("ranking")
    if not isinstance(ranking, list) or len(ranking) != 5:
        errors.append("ranking_must_have_5")
    for key in ("feasibility", "evidence", "confidence", "abstain"):
        if key not in payload:
            errors.append(f"missing_{key}")
    return (not errors), errors
