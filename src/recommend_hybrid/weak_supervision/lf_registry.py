"""Phase 2 LF registry and independent LF families."""
from __future__ import annotations

from pathlib import Path

import yaml

from .labels import LF_ABSTAIN, TargetLabel
from .lf_contracts import LabelingFunction

ROOT = Path(__file__).resolve().parents[3]


def _applicability(row: dict) -> int:
    return TargetLabel.INAPPROPRIATE if not _supported(row) else LF_ABSTAIN


def _supported(row: dict) -> bool:
    return row["dataset"] in row.get("action_datasets", ()) and row["stage"] in row.get("action_stages", ())


def _evidence(row: dict) -> int:
    return TargetLabel.CONDITIONAL if _supported(row) and len(row.get("missingness_flags", ())) else LF_ABSTAIN


def _published(row: dict) -> int:
    if not _supported(row): return TargetLabel.INAPPROPRIATE
    return TargetLabel.APPROPRIATE if row["action_status"] == "EVIDENCE_MAPPED" else LF_ABSTAIN


def _risk_uncertainty(row: dict) -> int:
    if not _supported(row): return LF_ABSTAIN
    if row["uncertainty"] >= 0.70: return TargetLabel.CONDITIONAL
    return TargetLabel.APPROPRIATE if row["prediction_risk"] >= 0.60 else LF_ABSTAIN


def _safety(row: dict) -> int:
    return TargetLabel.CONDITIONAL if _supported(row) and row["human_review_required"] else LF_ABSTAIN


def _uci_state(row: dict) -> int:
    if row["dataset"] not in {"student_mat", "student_por"} or not _supported(row): return LF_ABSTAIN
    if row.get("absences", 0) >= 6 or row.get("study_time", 9) <= 2 or row.get("previous_failures", 0) >= 1: return TargetLabel.APPROPRIATE
    if row.get("G1") is not None and row["G1"] < 12: return TargetLabel.APPROPRIATE
    if row.get("G2") is not None and row["G2"] < 12: return TargetLabel.APPROPRIATE
    return LF_ABSTAIN


def _oulad_state(row: dict) -> int:
    if row["dataset"] != "oulad" or not _supported(row): return LF_ABSTAIN
    if row.get("activity_level", 99) <= 10 or row.get("inactivity_streak", 0) >= 7: return TargetLabel.APPROPRIATE
    if row.get("recent_activity_trend", 0) < 0: return TargetLabel.APPROPRIATE
    return LF_ABSTAIN


def registry() -> tuple[LabelingFunction, ...]:
    all_datasets = ("student_mat", "student_por", "oulad")
    all_stages = ("S0", "S1", "S2", "EARLY_20", "EARLY_35", "MIDDLE_50", "LATE_75")
    all_actions = tuple()
    return (
        LabelingFunction("LF_ACTION_APPLICABILITY_V1", "ACTION_APPLICABILITY", all_datasets, all_stages, all_actions, ("action_datasets", "action_stages"), (), "Marks unsupported dataset/stage candidates inappropriate.", "phase2_v1", _applicability),
        LabelingFunction("LF_EVIDENCE_AVAILABILITY_V1", "EVIDENCE_AVAILABILITY", all_datasets, all_stages, all_actions, ("missingness_flags",), (), "Marks incomplete required evidence conditional without imputing missing values.", "phase2_v1", _evidence),
        LabelingFunction("LF_PUBLISHED_EVIDENCE_V1", "PUBLISHED_EVIDENCE", all_datasets, all_stages, all_actions, ("action_status",), ("WWC_STUDY_2007", "WWC_TECH_POSTSECONDARY_2019", "WWC_ADVISING_2021", "JISC_LEARNING_ANALYTICS_CODE_2023", "EEF_METACOGNITION_2026"), "Uses only Phase 1 action-evidence mapping; gaps abstain.", "phase2_v1", _published),
        LabelingFunction("LF_PREDICTION_UNCERTAINTY_V1", "PREDICTION_RISK_UNCERTAINTY", all_datasets, all_stages, all_actions, ("prediction_risk", "uncertainty"), (), "Frozen prediction is contextual; high uncertainty can only make a vote more conservative.", "phase2_v1", _risk_uncertainty),
        LabelingFunction("LF_HUMAN_REVIEW_SAFETY_V1", "HUMAN_REVIEW_SAFETY", all_datasets, all_stages, all_actions, ("human_review_required",), ("JISC_LEARNING_ANALYTICS_CODE_2023",), "Human-review actions remain conditional rather than autonomous.", "phase2_v1", _safety),
        LabelingFunction("LF_UCI_STATE_ACTION_V1", "UCI_STATE_ACTION_FIT", ("student_mat", "student_por"), ("S0", "S1", "S2"), all_actions, ("absences", "study_time", "previous_failures", "G1", "G2", "action_datasets", "action_stages"), (), "Uses only stage-available UCI state fields; G3 is never exposed.", "phase2_v1", _uci_state),
        LabelingFunction("LF_OULAD_STATE_ACTION_V1", "OULAD_STATE_ACTION_FIT", ("oulad",), ("EARLY_20", "EARLY_35", "MIDDLE_50", "LATE_75"), all_actions, ("activity_level", "recent_activity_trend", "inactivity_streak", "action_datasets", "action_stages"), (), "Uses only cutoff-safe OULAD activity evidence.", "phase2_v1", _oulad_state),
    )


def write_registry(path: Path) -> None:
    payload = {"schema_version": "recommend_lf_registry_v1", "labeling_functions": [{"lf_id": lf.lf_id, "lf_family": lf.lf_family, "supported_datasets": list(lf.supported_datasets), "supported_stages": list(lf.supported_stages), "supported_action_ids": list(lf.supported_action_ids), "required_fields": list(lf.required_fields), "source_ids": list(lf.source_ids), "possible_outputs": [-1, 0, 1, 2], "rationale": lf.rationale, "version": lf.version} for lf in registry()]}
    path.write_text(yaml.safe_dump(payload, sort_keys=False, allow_unicode=True), encoding="utf-8")


__all__ = ["registry", "write_registry"]
