"""Panel C case construction and protocol helpers. No provider calls."""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import pandas as pd

from src.recommend_hybrid.v3.contracts import CanonicalAction
from src.recommend_hybrid.v3.feasibility import evaluate_action
from src.recommend_hybrid.v3.features_io import features_from_row

ROOT = Path(__file__).resolve().parents[3]
V3 = ROOT / "artifacts" / "recommend_hybrid" / "v3"
PROMPT_PATH = ROOT / "configs" / "recommend_hybrid" / "v3" / "panel_c_external_reviewer_v3.txt"
PROMPT_VERSION = "panel_c_external_reviewer_v3_c0_blinded_v1"
PROVIDER = "Google Gemini API"
MODEL_NAME = "gemini-3.5-flash-lite"
FORBIDDEN_PROMPT_TOKENS = (
    "risk_probability",
    "predicted_risk",
    "risk_threshold",
    "prediction_threshold",
    "risk_margin",
    "risk_band",
    "uncertainty",
    "uncertainty_band",
    "five-ebm",
    "five_ebm",
    "model_id",
    "model_name",
    "hybrid c0",
    "final_result",
    "seed_disagreement",
    "label_conflict",
    "query_id",
    "student_key",
    "id_student",
    "record_id",
)
ACTION_TITLES = {
    "ASSESSMENT_COMPLETION": "Complete missing or soon-due assessments before the next review.",
    "RECOVER_ENGAGEMENT": "Restore recent VLE participation after low activity.",
    "STUDY_REGULARITY": "Stabilize study cadence when the recent pattern is irregular.",
    "TARGETED_CONTENT_REVIEW": "Review unread or under-covered module materials.",
    "QUIZ_RETRIEVAL_PRACTICE": "Practice retrieval using available quizzes.",
}
EVIDENCE_FIELDS = (
    "assessments_due",
    "missing_assessment_count",
    "due_soon_count",
    "completion_rate",
    "time_to_deadline_days",
    "inactivity_streak",
    "active_day_rate",
    "recent_activity_trend",
    "regularity_score",
    "content_coverage",
    "quiz_activity",
    "assessment_window_open",
    "knowledge_gap_evidence",
)


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def canonical_json_bytes(value) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


def prompt_text() -> str:
    return PROMPT_PATH.read_text(encoding="utf-8").replace("\r\n", "\n").replace("\r", "\n")


def prompt_sha256() -> str:
    return sha256_bytes(prompt_text().encode("utf-8"))


def case_id_for_query(query_id: str) -> str:
    return "case_" + hashlib.sha256(str(query_id).encode("utf-8")).hexdigest()[:24]


def _jsonable(value):
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    if hasattr(value, "item"):
        try:
            value = value.item()
        except (ValueError, AttributeError):
            pass
    if isinstance(value, (bool, int, float, str)):
        return value
    if pd.isna(value):
        return None
    return value


def build_case_payload(row: pd.Series) -> tuple[str, dict, list[dict]]:
    features = features_from_row(row)
    evaluations = []
    candidates = []
    for action in CanonicalAction:
        result = evaluate_action(action, features)
        evaluations.append(
            {
                "action_id": action.value,
                "eligible": bool(result.eligible),
                "reason_codes": list(result.reason_codes),
            }
        )
        if result.eligible:
            candidates.append(
                {
                    "action_id": action.value,
                    "title": ACTION_TITLES[action.value],
                    "feasibility_reasons": list(result.reason_codes),
                }
            )
    evidence = {field: _jsonable(row.get(field)) for field in EVIDENCE_FIELDS}
    payload = {
        "stage": str(row["stage"]),
        "cutoff_day": int(row["cutoff_day"]),
        "course_progress": _jsonable(row.get("course_progress")),
        "observed_pre_cutoff_evidence": evidence,
        "availability": {
            "vle_access_available": _jsonable(row.get("vle_access_available")),
            "study_material_available": _jsonable(row.get("study_material_available")),
            "quiz_available": _jsonable(row.get("quiz_available")),
        },
        "candidate_actions": candidates,
    }
    return case_id_for_query(str(row["query_id"])), payload, evaluations


def forbidden_hits(text: str) -> list[str]:
    lowered = text.lower()
    hits = []
    for token in FORBIDDEN_PROMPT_TOKENS:
        if token.lower() in lowered:
            hits.append(token)
    return hits


def assert_payload_blinded(payload: dict, prompt: str) -> None:
    blob = prompt + "\n" + json.dumps(payload, ensure_ascii=False)
    hits = forbidden_hits(blob)
    if hits:
        raise ValueError(f"Panel C payload/prompt contains forbidden tokens: {hits}")


def load_panel_c_feature_rows() -> pd.DataFrame:
    cases = pd.read_parquet(V3 / "panel_c" / "PANEL_C_SAMPLED_CASES.parquet")
    features = pd.read_parquet(V3 / "data" / "learner_stage_features.parquet")
    merged = cases.merge(features, on=["query_id", "student_key", "course_key", "stage", "cutoff_day"], how="left", suffixes=("", "_feat"))
    if merged["record_id"].isna().any():
        raise RuntimeError("Panel C sample is missing feature rows")
    if merged.query_id.duplicated().any():
        raise RuntimeError("duplicate Panel C query_id")
    return merged.sort_values("query_id").reset_index(drop=True)
