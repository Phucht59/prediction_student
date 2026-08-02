"""Leakage-safe Phase 4 feature construction."""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
from src.recommend_hybrid.weak_supervision.registry import load_action_mappings

ROOT = Path(__file__).resolve().parents[3]
BLACKLIST = {"silver_label", "silver_status", "silver_confidence", "silver_prob_0", "silver_prob_1", "silver_prob_2", "silver_expected_relevance", "lf_votes", "lf_conflict", "lf_family_coverage", "split", "query_id", "student_key", "lineage_hash", "target", "outer_label", "final_result", "G3", "post_cutoff_event", "source_id", "evidence_source_ids", "action_status"}
NUMERIC = ("prediction_risk", "uncertainty", "absences", "study_time", "previous_failures", "G1", "G2", "activity_level", "recent_activity_trend", "inactivity_streak", "assessment_progress", "assessments_due", "grade_trend", "knowledge_gap", "course_progress")


def validate_feature_leakage(columns):
    bad = BLACKLIST & set(columns)
    if bad: raise ValueError("PHASE4_FEATURE_LEAKAGE_DETECTED: " + ",".join(sorted(bad)))


def build_schema(frame: pd.DataFrame):
    actions = {a.action_id: a for a in load_action_mappings(ROOT / "artifacts/recommend_hybrid/scientific_labeling/action_evidence_map.yaml")}
    return {"numeric": list(NUMERIC), "feature_dim": len(NUMERIC) * 2,
            "action_ids": sorted(actions), "datasets": sorted(frame.dataset.unique()),
            "stages": sorted(frame.stage.unique()), "blacklisted": sorted(BLACKLIST),
            "prediction_authority_required": "Hybrid CNN-BiLSTM"}


def prepare(frame: pd.DataFrame, schema=None):
    schema = schema or build_schema(frame)
    validate_feature_leakage(schema["numeric"])
    actions = {a.action_id: a for a in load_action_mappings(ROOT / "artifacts/recommend_hybrid/scientific_labeling/action_evidence_map.yaml")}
    x = frame.reindex(columns=schema["numeric"], fill_value=np.nan).apply(pd.to_numeric, errors="coerce")
    missing = x.isna().astype("float32")
    x = x.fillna(0).astype("float32")
    lookup = lambda values, names: np.asarray([names.index(v) for v in values], dtype=np.int64)
    action = lookup(frame.action_id, schema["action_ids"]); dataset = lookup(frame.dataset, schema["datasets"]); stage = lookup(frame.stage, schema["stages"])
    meta = np.asarray([[actions[a].estimated_minutes / 180, {"LOW": .25, "MEDIUM": .5, "HIGH": .75, "CRITICAL": 1}.get(actions[a].default_priority, .5), float(actions[a].human_review_required)] for a in frame.action_id], dtype=np.float32)
    return np.concatenate((x.to_numpy(), missing.to_numpy()), axis=1), action, dataset, stage, meta, schema
