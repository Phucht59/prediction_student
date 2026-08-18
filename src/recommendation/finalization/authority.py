"""Deterministic freeze-authority paths and checksum validation."""

from __future__ import annotations

import json
from pathlib import Path

from ..models.features import APPROVED_FEATURES
from ..weak_supervision.matrix import FINAL_ACTIONS
from ..weak_supervision.silver import sha256_file
from . import FREEZE_VERSION

ACTION_STATUS = {
    "assessment_recovery": "PASS",
    "re_engagement": "PASS",
    "study_planning": "PASS",
    "progress_monitoring": "PASS_WITH_WARNING",
    "retrieval_practice": "REVIEW",
}
ACTION_DISPLAY = {
    "assessment_recovery": "Assessment Recovery",
    "re_engagement": "Re-engagement",
    "study_planning": "Study Planning",
    "progress_monitoring": "Progress Monitoring",
    "retrieval_practice": "Retrieval Practice",
}
RETIRED_ACTIONS = ("content_review",)
REJECTED_ACTIONS = ("academic_help_seeking",)
EBM_KEYS = {
    "assessment_recovery": "A1",
    "re_engagement": "A2",
    "study_planning": "A3",
    "progress_monitoring": "A4",
    "retrieval_practice": "A5",
}

REQUIRED_RELATIVE = {
    "phase6_source_manifest": "artifacts/recommendation/labeling/phase6_source_manifest.json",
    "phase7_manifest": "artifacts/recommendation/weak_supervision/phase7_manifest.json",
    "silver_labels": "artifacts/recommendation/weak_supervision/silver_labels.parquet",
    "phase8_model_manifest": "artifacts/recommendation/models/phase8_model_manifest.json",
    "phase9_manifest": "artifacts/recommendation/evaluation/phase9_manifest.json",
    "panel_b_reference": "artifacts/recommendation/evaluation/panel_b_reference.parquet",
    "panel_b_metrics": "artifacts/recommendation/evaluation/panel_b_metrics.json",
    "panel_b_bootstrap": "artifacts/recommendation/evaluation/panel_b_bootstrap.parquet",
    "feasibility_v2_config": "configs/recommendation/feasibility_v2.yaml",
    "actions_config": "configs/recommendation/actions.yaml",
    "phase8_config": "configs/recommendation/phase8.yaml",
    "phase9_config": "configs/recommendation/phase9.yaml",
}
OPTIONAL_RELATIVE = {
    "scores": "artifacts/recommendation/final/oulad_recommendation_scores.parquet",
    "plans": "artifacts/recommendation/final/oulad_recommendation_plans.parquet",
}


def ebm_paths() -> dict[str, str]:
    return {
        action: f"artifacts/recommendation/models/ebm/{key}_ebm.pkl"
        for action, key in EBM_KEYS.items()
    }


def baseline_paths() -> dict[str, dict[str, str]]:
    kinds = {"ACTION_STAGE_PRIOR": "stage_prior", "RIDGE": "ridge", "RANDOM_FOREST": "rf"}
    return {
        action: {name: f"artifacts/recommendation/models/baselines/{key}_{suffix}.pkl" for name, suffix in kinds.items()}
        for action, key in EBM_KEYS.items()
    }


def checksum_map(root: Path, relatives: dict[str, str]) -> dict[str, str]:
    return {name: sha256_file(root / relative) for name, relative in relatives.items() if (root / relative).exists()}


def load_json(root: Path, relative: str) -> dict:
    return json.loads((root / relative).read_text(encoding="utf-8"))


def validate_required_artifacts(root: Path) -> list[str]:
    blockers = []
    for name, relative in REQUIRED_RELATIVE.items():
        if not (root / relative).exists():
            blockers.append(f"missing:{relative}")
    for action, relative in ebm_paths().items():
        if not (root / relative).exists():
            blockers.append(f"missing_ebm:{action}")
    return blockers


def validate_scientific_authority(root: Path) -> list[str]:
    blockers = []
    phase8 = load_json(root, REQUIRED_RELATIVE["phase8_model_manifest"])
    phase9 = load_json(root, REQUIRED_RELATIVE["phase9_manifest"])
    phase7 = load_json(root, REQUIRED_RELATIVE["phase7_manifest"])
    if list(phase8.get("features") or []) != list(APPROVED_FEATURES):
        blockers.append("feature_contract_mismatch")
    if set(phase8.get("models") or {}) != set(FINAL_ACTIONS):
        blockers.append("phase8_action_taxonomy_mismatch")
    if phase8["models"]["retrieval_practice"]["quality_status"] != "REVIEW":
        blockers.append("a5_review_flag_missing")
    if "PASS_WITH_WARNING" not in str(phase8["models"]["progress_monitoring"]["quality_status"]):
        blockers.append("a4_warning_missing")
    if phase8.get("course_progress") != "FEATURE_EXCLUDED_REDUNDANT_STAGE":
        blockers.append("course_progress_not_excluded")
    a4 = phase8.get("a4_feasibility_audit") or {}
    if (a4.get("new_rule") or {}).get("status") != "FEASIBLE":
        blockers.append("a4_old_content_rule_active")
    if (a4.get("old_rule") or {}).get("status") != "UNKNOWN":
        blockers.append("a4_old_rule_not_preserved")
    if phase9.get("evaluation_name") != "AUTOMATED_REFERENCE_EVALUATION":
        blockers.append("phase9_not_automated_reference")
    if phase9.get("panel_b_overlap_with_training") not in {0, None}:
        blockers.append("panel_b_leakage")
    if phase9.get("models_tuned_on_panel_b") is True:
        blockers.append("panel_b_used_for_tuning")
    if set(phase7.get("actions") or []) != set(FINAL_ACTIONS):
        blockers.append("phase7_action_taxonomy_mismatch")
    return blockers


def validate_checksums(root: Path, freeze: dict) -> list[str]:
    blockers = []
    expected = freeze.get("content", {}).get("checksums", {})
    current = checksum_map(root, {**REQUIRED_RELATIVE, **{f"ebm_{action}": path for action, path in ebm_paths().items()}})
    for name, digest in expected.items():
        if name.startswith("optional_"):
            continue
        if name not in current:
            blockers.append(f"checksum_missing_file:{name}")
        elif current[name] != digest:
            blockers.append(f"checksum_mismatch:{name}")
    for action, relative in ebm_paths().items():
        key = f"ebm_{action}"
        if key in expected and expected[key] != sha256_file(root / relative):
            blockers.append(f"ebm_modified_after_freeze:{action}")
    return blockers


def freeze_content(root: Path) -> dict:
    relatives = dict(REQUIRED_RELATIVE)
    relatives.update({f"ebm_{action}": path for action, path in ebm_paths().items()})
    optional = checksum_map(root, OPTIONAL_RELATIVE)
    return {
        "freeze_version": FREEZE_VERSION,
        "actions": list(FINAL_ACTIONS),
        "action_status": ACTION_STATUS,
        "action_display": ACTION_DISPLAY,
        "retired_actions": list(RETIRED_ACTIONS),
        "rejected_actions": list(REJECTED_ACTIONS),
        "features": list(APPROVED_FEATURES),
        "artifacts": relatives,
        "optional_artifacts": OPTIONAL_RELATIVE,
        "baseline_artifacts": baseline_paths(),
        "checksums": checksum_map(root, relatives),
        "optional_checksums": optional,
        "metric_contract": {
            "primary": "NDCG@3",
            "evaluation_name": "AUTOMATED_REFERENCE_EVALUATION",
            "disclaimer": "Panel B references are automated Gemini-family labels, not expert ground truth and not causal efficacy.",
        },
    }
