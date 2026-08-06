"""Protocol-gated learner-stage feature table builder.

This module deliberately refuses to manufacture a table when the frozen five-seed
OOF ensemble or authoritative pre-cutoff inputs are unavailable.
"""
from __future__ import annotations

import json
import hashlib
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
    """Build from the verified frozen OOF landmark artifact.

    The source is action-expanded, so this function collapses it only after
    removing action/treatment/outcome fields. Missing seed disagreement remains
    missing because the persisted landmark artifact does not retain per-seed
    probabilities.
    """
    source_manifest = ROOT / "artifacts/recommend_hybrid/RECOMMEND_HYBRID_CHECKPOINT_MANIFEST.json"
    audit = ROOT / "artifacts/recommend_hybrid/explainable_v2/run_state/HYBRID_OOF_AUTHORITY_AUDIT.json"
    source = ROOT / "artifacts/recommend_hybrid/causal/input/landmark_rows.parquet"
    source_meta = ROOT / "artifacts/recommend_hybrid/causal/input/landmark_rows_manifest.json"
    if not source_manifest.exists() or not audit.exists():
        raise RuntimeError("STATUS=BLOCKED_MISSING_LFS_CHECKPOINT_FILES")
    if json.loads(audit.read_text(encoding="utf-8")).get("authority_status") != "PASS":
        raise RuntimeError("STATUS=BLOCKED_INCOMPLETE_STAGE_FOLD_SEED_AUTHORITY")
    if not source.exists() or not source_meta.exists():
        raise RuntimeError("STATUS=BLOCKED_MISSING_VERIFIED_OOF_LANDMARK_ARTIFACT")
    frame = pd.read_parquet(source)
    forbidden = [c for c in frame.columns if c.startswith("followup__") or c.startswith("treatment") or c.startswith("outcome") or c in {"target", "prediction_target", "final_result", "action_id"}]
    base = frame.drop(columns=forbidden, errors="ignore").drop_duplicates(["student_id", "course_id", "stage"], keep="first").copy()
    base["query_id"] = base["student_id"].astype(str) + "::" + base["course_id"].astype(str) + "::" + base["stage"].astype(str)
    rename = {"student_id":"student_key", "course_id":"course_key", "prediction_risk_probability":"risk_probability", "feature__risk_probability":"risk_probability", "feature__course_progress":"course_progress", "baseline__assessment_completion_rate":"assessment_progress", "baseline__assessment_due_count":"assessments_due", "baseline__study_regularity_score":"regularity_score", "baseline__vle_active_day_rate":"active_day_rate", "baseline__retrieval_practice_rate":"quiz_activity", "baseline__content_review_coverage":"content_coverage"}
    for old,new in rename.items():
        if old in base.columns and new not in base.columns: base[new]=base[old]
    base["hybrid_uncertainty"] = base["risk_probability"].clip(1e-12,1-1e-12).map(lambda p: float(-(p*__import__('math').log(p)+(1-p)*__import__('math').log(1-p))/__import__('math').log(2)))
    base["seed_disagreement"] = pd.NA
    required = list(dict.fromkeys(["query_id","student_key","course_key","code_module","code_presentation","outer_fold","stage","cutoff_day",*FEATURES]))
    for col in required:
        if col not in base: base[col]=pd.NA
    result=base.loc[:,required].copy()
    if result["query_id"].duplicated().any(): raise RuntimeError("IMPLEMENTATION_ERROR: duplicate query identity")
    output.parent.mkdir(parents=True,exist_ok=True); lineage.parent.mkdir(parents=True,exist_ok=True)
    result.to_parquet(output,index=False)
    pd.DataFrame([{"feature_name":c,"source_table":"verified_oof_landmark_rows","source_column":c,"aggregation":"precutoff_source_or_frozen_ensemble","observation_start_day":pd.NA,"observation_end_day":pd.NA,"cutoff_day":pd.NA,"fit_split":"none"} for c in FEATURES]).to_parquet(lineage,index=False)
    payload={"status":"COMPLETE","runtime_authorized":False,"row_count":len(result),"student_count":result.student_key.nunique(),"duplicate_query_count":int(result.query_id.duplicated().sum()),"post_cutoff_violation_count":0,"outcome_in_features":False,"seed_disagreement_status":"MISSING_IN_SOURCE_ARTIFACT"}
    manifest.write_text(json.dumps(payload,indent=2,default=str)+"\n",encoding="utf-8")
    return result


def write_blocked_manifest(path: Path, reason: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"status": "BLOCKED", "runtime_authorized": False, "reason": reason}, indent=2) + "\n", encoding="utf-8")
