"""Build freeze manifest and thesis source-of-truth from existing artifacts only."""

from __future__ import annotations

import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

from ..weak_supervision.silver import write_json
from . import BUNDLE_VERSION, FREEZE_VERSION, STATE_VERSION
from .authority import REQUIRED_RELATIVE, freeze_content, load_json


def git_metadata(root: Path) -> dict:
    def _run(args: list[str]) -> str:
        result = subprocess.run(args, cwd=root, capture_output=True, text=True, check=False)
        return result.stdout.strip() if result.returncode == 0 else "UNKNOWN"

    status = _run(["git", "status", "--porcelain"])
    return {
        "branch": _run(["git", "rev-parse", "--abbrev-ref", "HEAD"]),
        "head": _run(["git", "rev-parse", "HEAD"]),
        "working_tree_dirty": bool(status),
    }


def source_of_truth(root: Path) -> dict:
    phase6 = load_json(root, REQUIRED_RELATIVE["phase6_source_manifest"])
    phase7 = load_json(root, REQUIRED_RELATIVE["phase7_manifest"])
    phase8 = load_json(root, REQUIRED_RELATIVE["phase8_model_manifest"])
    phase9 = load_json(root, REQUIRED_RELATIVE["phase9_manifest"])
    models = {}
    for action, item in phase8["models"].items():
        models[action] = {
            "training_rows": item["training_rows"],
            "excluded_no_evidence_rows": item["excluded_no_evidence_rows"],
            "oof_mae": item["cv_metrics"]["mae"],
            "oof_rmse": item["cv_metrics"]["rmse"],
            "selected_config": {
                "max_bins": item["selected_config"]["max_bins"],
                "interactions": item["selected_config"]["interactions"],
                "min_samples_leaf": item["selected_config"]["min_samples_leaf"],
            },
            "quality_status": item["quality_status"],
            "checksum": item["checksum"],
        }
    return {
        "freeze_version": FREEZE_VERSION,
        "bundle_version": BUNDLE_VERSION,
        "state_version": STATE_VERSION,
        "final_actions": list(phase7["actions"]),
        "action_status": {action: item["quality_status"] for action, item in phase8["models"].items()},
        "phase6": {
            "version": phase6.get("version"),
            "effective_llm_rows": phase6.get("effective_llm_rows"),
            "behavioral_rows": phase6.get("behavioral_rows"),
            "panel_a_case_count": phase6.get("panel_a_case_count"),
            "panel_b_case_count": phase6.get("panel_b_case_count"),
            "panel_b_overlap_count": phase6.get("panel_b_overlap_count"),
        },
        "phase7": {
            "silver_status_counts": phase7.get("silver_status_counts"),
            "aggregator_by_action": phase7.get("aggregator_by_action"),
            "valid_counts": phase7.get("valid_counts"),
            "review_counts": phase7.get("review_counts"),
            "no_evidence_counts": phase7.get("no_evidence_counts"),
        },
        "phase8": {
            "features": phase8.get("features"),
            "course_progress": phase8.get("course_progress"),
            "models": models,
            "panel_a_oof_ranking": phase8.get("panel_a_oof_ranking"),
        },
        "phase9": {
            "evaluation_name": phase9.get("evaluation_name"),
            "reference_status_counts": phase9.get("reference_status_counts"),
            "agreement": phase9.get("agreement"),
            "metrics": phase9.get("metrics"),
            "bootstrap": phase9.get("bootstrap"),
            "a5": phase9.get("a5"),
            "panel_b_overlap_with_training": phase9.get("panel_b_overlap_with_training"),
            "models_tuned_on_panel_b": phase9.get("models_tuned_on_panel_b"),
        },
        "warnings": {
            "a4": "Gemini 3.5 and Gemini 3.1 are same-family weak sources, not independent experts. Feasibility v2 marks Progress Monitoring FEASIBLE.",
            "a5": "A5 remains REVIEW due to high weak-source conflict and is not suppressed at runtime.",
            "reference": "Panel B evaluation is AUTOMATED_REFERENCE_EVALUATION, not expert ground truth and not causal efficacy.",
        },
    }


def write_freeze_artifacts(root: Path) -> dict:
    content = freeze_content(root)
    metadata = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "git": git_metadata(root),
    }
    freeze = {"content": content, "metadata": metadata, "freeze_version": FREEZE_VERSION}
    write_json(root / "artifacts/recommendation/final/FINAL_RECOMMENDATION_FREEZE_MANIFEST.json", freeze)
    truth = source_of_truth(root)
    write_json(root / "artifacts/recommendation/final/THESIS_RECOMMENDATION_SOURCE_OF_TRUTH.json", truth)
    return freeze
