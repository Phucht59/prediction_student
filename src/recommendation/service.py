"""Frozen runtime RecommendationService. No training, Snorkel, or LLM."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from .finalization import BUNDLE_VERSION, FREEZE_VERSION, STATE_VERSION
from .finalization.authority import ACTION_DISPLAY, ACTION_STATUS, ebm_paths, validate_checksums, validate_required_artifacts, validate_scientific_authority
from .finalization.freeze import write_freeze_artifacts
from .models.ebm import load_model
from .models.features import APPROVED_FEATURES, encode_state_features
from .ranking.explain import compact_explanation
from .ranking.ranker import rank_actions, top_k_actions
from .ranking.scorer import ACTION_QUALITY, score_case
from .weak_supervision.matrix import FINAL_ACTIONS
from .weak_supervision.silver import sha256_file

FORBIDDEN_INPUT = {"final_result", "target", "future_vle", "future_assessment", "future_unregistration", "prediction_truth_label"}
REQUIRED_STATE = (
    "stage",
    "risk_probability",
    "inactive_streak",
    "active_days_ratio",
    "recent_activity",
    "activity_trend",
    "assessment_completion",
    "missing_assessments",
    "quiz_activity",
    "vle_available",
)


class RecommendationService:
    def __init__(self, root: Path, *, persist: bool = False, freeze: dict | None = None):
        self.root = Path(root)
        blockers = validate_required_artifacts(self.root) + validate_scientific_authority(self.root)
        if blockers:
            raise RuntimeError(f"frozen authority failed: {blockers}")
        self.freeze = freeze or write_freeze_artifacts(self.root)
        checksum_blockers = validate_checksums(self.root, self.freeze)
        if checksum_blockers:
            raise RuntimeError(f"model bundle checksum failed: {checksum_blockers}")
        if list(self.freeze["content"]["features"]) != list(APPROVED_FEATURES):
            raise RuntimeError("feature contract mismatch")
        self.models = {}
        for action, relative in ebm_paths().items():
            path = self.root / relative
            expected = self.freeze["content"]["checksums"][f"ebm_{action}"]
            if sha256_file(path) != expected:
                raise RuntimeError(f"EBM checksum mismatch for {action}")
            self.models[action] = {
                "model": load_model(path),
                "version": BUNDLE_VERSION,
                "quality_status": ACTION_STATUS[action],
            }
        if set(self.models) != set(FINAL_ACTIONS):
            raise RuntimeError("bundle must contain exactly five EBM models")
        self.persist_enabled = persist
        self.bundle_version = BUNDLE_VERSION
        self.state_version = STATE_VERSION

    def health(self) -> dict:
        database = "disabled"
        if self.persist_enabled:
            try:
                from src.database.connection import test_connection

                test_connection()
                database = "ok"
            except Exception:
                database = "unavailable"
        return {
            "status": "ok" if database != "unavailable" else "degraded",
            "database": database,
            "model_bundle": "ok",
            "freeze_version": FREEZE_VERSION,
        }

    @staticmethod
    def validate_state(payload: dict) -> list[str]:
        errors = []
        leaked = FORBIDDEN_INPUT.intersection(payload)
        if leaked:
            errors.append(f"forbidden_fields:{sorted(leaked)}")
        missing = [field for field in REQUIRED_STATE if field not in payload]
        if missing:
            errors.append(f"missing_fields:{missing}")
        stage = str(payload.get("stage", ""))
        if stage and stage not in {"20pct", "35pct", "50pct", "75pct"}:
            errors.append("invalid_stage")
        if "FINAL" in stage:
            errors.append("final_stage_not_allowed")
        risk = payload.get("risk_probability")
        if risk is not None:
            try:
                if not 0 <= float(risk) <= 1:
                    errors.append("risk_probability_out_of_range")
            except (TypeError, ValueError):
                errors.append("risk_probability_not_numeric")
        return errors

    def recommend(self, payload: dict, *, persist: bool | None = None) -> dict:
        errors = self.validate_state(payload)
        if errors:
            raise ValueError({"validation_errors": errors})
        row = pd.Series(payload)
        if "case_id" not in row or pd.isna(row.get("case_id")):
            row["case_id"] = str(payload.get("enrollment_identity") or payload.get("record_id") or "anonymous")
        ranked = rank_actions(score_case(row, self.models, model_version=self.bundle_version), top_k=3)
        result = self._format(row, ranked)
        should_persist = self.persist_enabled if persist is None else persist
        if should_persist:
            from .persistence.runtime import persist_recommendation

            result["run_id"] = persist_recommendation(result, payload, bundle_version=self.bundle_version, state_version=self.state_version)
        return result

    def _format(self, row: pd.Series, ranked: list[dict]) -> dict:
        actions = []
        for item in ranked:
            explanation = compact_explanation(item)
            actions.append({
                "action_id": item["action_id"],
                "display_name": ACTION_DISPLAY[item["action_id"]],
                "raw_score": item["raw_score"],
                "relevance_score": item["relevance_score"],
                "rank": item["rank"],
                "releasable_rank": item.get("releasable_rank"),
                "feasibility_status": item["feasibility_status"],
                "release_status": item["release_status"],
                "quality_warning": item.get("quality_warning") or ACTION_QUALITY[item["action_id"]],
                "in_top_k": item.get("in_top_k"),
                "top_positive_reasons": explanation["top_positive_reasons"],
                "top_negative_reasons": explanation["top_negative_reasons"],
                "intercept": item.get("intercept"),
                "model_version": item.get("model_version"),
            })
        return {
            "case_id": str(row.get("case_id")),
            "enrollment_identity": str(row.get("enrollment_identity") or row.get("record_id") or ""),
            "student_id": str(row.get("student_id") or ""),
            "module": str(row.get("module") or ""),
            "presentation": str(row.get("presentation") or ""),
            "stage": str(row.get("stage")),
            "risk_probability": float(row.get("risk_probability")),
            "plan_status": ranked[0]["plan_status"],
            "top_actions": [action for action in actions if action.get("in_top_k")][:3],
            "actions": actions,
            "bundle_version": self.bundle_version,
            "state_version": self.state_version,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "wording": "recommended support action / relevance score / model explanation; not a guaranteed improvement",
        }


def encode_or_raise(frame: pd.DataFrame) -> pd.DataFrame:
    return encode_state_features(frame)
