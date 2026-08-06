"""Protocol-gated learner-stage feature table builder.

This module deliberately refuses to manufacture a table when the frozen five-seed
OOF ensemble or authoritative pre-cutoff inputs are unavailable.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
FEATURES = (
    "risk_probability", "hybrid_uncertainty", "seed_disagreement", "course_progress",
    "assessment_progress", "assessments_due", "time_to_deadline_days",
    "inactivity_streak", "active_day_rate", "recent_activity_trend",
    "regularity_score", "content_coverage", "quiz_activity", "stage",
)


def build(output: Path, lineage: Path, manifest: Path) -> pd.DataFrame:
    """Build only from a validated source; never substitutes old action rows."""
    source_manifest = ROOT / "artifacts/recommend_hybrid/RECOMMEND_HYBRID_CHECKPOINT_MANIFEST.json"
    if not source_manifest.exists():
        raise RuntimeError("frozen Hybrid checkpoint manifest is missing")
    raise RuntimeError(
        "BLOCKED: authoritative five-seed OOF learner-stage feature source is not available; "
        "no synthetic or single-seed substitute was used"
    )


def write_blocked_manifest(path: Path, reason: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"status": "BLOCKED", "runtime_authorized": False, "reason": reason}, indent=2) + "\n", encoding="utf-8")
