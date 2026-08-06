"""Build learner stage feature table and candidate action feature table."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.recommend_hybrid.explainable_v2.contracts import CanonicalAction
from src.recommend_hybrid.explainable_v2.data_builder import build as build_learner_features, write_blocked_manifest


def build_candidate_action_table(
    learner_features_df: pd.DataFrame, output_path: Path
) -> pd.DataFrame:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    candidate_rows = []

    actions = [a.value for a in CanonicalAction]

    for row in learner_features_df.itertuples(index=False):
        row_dict = row._asdict()
        student_key = str(row_dict.get("student_key", ""))
        course_key = str(row_dict.get("course_key", ""))
        stage = str(row_dict.get("stage", ""))
        query_id = str(row_dict.get("query_id", f"{student_key}::{course_key}::{stage}"))
        case_id = f"case_{hash((student_key, course_key, stage)) & 0xffffffffffff:012x}"

        risk_prob = float(row_dict.get("risk_probability", 0.3)) if pd.notna(row_dict.get("risk_probability")) else 0.3
        inactivity = int(row_dict.get("inactivity_streak", 0)) if pd.notna(row_dict.get("inactivity_streak")) else 0
        due_count = int(row_dict.get("assessments_due", 0)) if pd.notna(row_dict.get("assessments_due")) else 0
        regularity = float(row_dict.get("regularity_score", 0.5)) if pd.notna(row_dict.get("regularity_score")) else 0.5
        content_cov = float(row_dict.get("content_coverage", 0.5)) if pd.notna(row_dict.get("content_coverage")) else 0.5
        quiz_act = float(row_dict.get("quiz_activity", 0.5)) if pd.notna(row_dict.get("quiz_activity")) else 0.5

        for act in actions:
            cand = {
                "query_id": query_id,
                "case_id": case_id,
                "student_group_id": student_key,
                "outer_fold": int(row_dict.get("outer_fold", 0)) if pd.notna(row_dict.get("outer_fold")) else 0,
                "stage": stage,
                "action_id": act,
                # Shared features
                "risk_probability": risk_prob,
                "risk_band": "HIGH" if risk_prob > 0.6 else ("BORDERLINE" if risk_prob > 0.3 else "LOW"),
                "hybrid_uncertainty": float(row_dict.get("hybrid_uncertainty", 0.1)) if pd.notna(row_dict.get("hybrid_uncertainty")) else 0.1,
                "seed_disagreement": float(row_dict.get("seed_disagreement", 0.05)) if pd.notna(row_dict.get("seed_disagreement")) else 0.05,
                "inactivity_streak": inactivity,
                "active_day_rate": float(row_dict.get("active_day_rate", 0.5)) if pd.notna(row_dict.get("active_day_rate")) else 0.5,
                "recent_activity_trend": float(row_dict.get("recent_activity_trend", 0.0)) if pd.notna(row_dict.get("recent_activity_trend")) else 0.0,
                "assessments_due": due_count,
                "regularity_score": regularity,
                "content_coverage": content_cov,
                "quiz_activity": quiz_act,
                # Action-specific evidence features
                "missing_assessment_count": due_count if act == CanonicalAction.ASSESSMENT_COMPLETION.value else 0,
                "due_soon_count": due_count if act == CanonicalAction.ASSESSMENT_COMPLETION.value else 0,
                "completion_rate": float(row_dict.get("assessment_progress", 0.5)) if pd.notna(row_dict.get("assessment_progress")) else 0.5,
                "recent_activity_drop": -0.3 if (act == CanonicalAction.RECOVER_ENGAGEMENT.value and inactivity > 3) else 0.0,
                "engagement_recovery_possible": True if act == CanonicalAction.RECOVER_ENGAGEMENT.value else False,
                "inter_session_gap": inactivity * 24.0 if act == CanonicalAction.STUDY_REGULARITY.value else 0.0,
                "study_consistency": regularity if act == CanonicalAction.STUDY_REGULARITY.value else 0.0,
                "unviewed_content": (1.0 - content_cov) if act == CanonicalAction.TARGETED_CONTENT_REVIEW.value else 0.0,
                "low_coverage_topics": 2 if (act == CanonicalAction.TARGETED_CONTENT_REVIEW.value and content_cov < 0.5) else 0,
                "quiz_available": True if act == CanonicalAction.QUIZ_RETRIEVAL_PRACTICE.value else False,
                "low_quiz_score": True if (act == CanonicalAction.QUIZ_RETRIEVAL_PRACTICE.value and quiz_act < 0.4) else False,
            }
            candidate_rows.append(cand)

    df_candidates = pd.DataFrame(candidate_rows)
    df_candidates.to_parquet(output_path, index=False)
    return df_candidates


def main() -> int:
    root = ROOT
    p = argparse.ArgumentParser()
    p.add_argument(
        "--output",
        type=Path,
        default=root / "artifacts/recommend_hybrid/explainable_v2/data/learner_stage_features.parquet",
    )
    p.add_argument(
        "--candidates-output",
        type=Path,
        default=root / "artifacts/recommend_hybrid/explainable_v2/features/action_candidates.parquet",
    )
    p.add_argument(
        "--lineage",
        type=Path,
        default=root / "artifacts/recommend_hybrid/explainable_v2/data/feature_lineage.parquet",
    )
    p.add_argument(
        "--manifest",
        type=Path,
        default=root / "artifacts/recommend_hybrid/explainable_v2/data/FEATURE_TABLE_MANIFEST.json",
    )
    a = p.parse_args()
    try:
        df_learner = build_learner_features(a.output, a.lineage, a.manifest)
        build_candidate_action_table(df_learner, a.candidates_output)
        print(f"BUILD_FEATURE_TABLE_SUCCESS=TRUE, ROWS={len(df_learner)}")
    except RuntimeError as exc:
        write_blocked_manifest(a.manifest, str(exc))
        print(str(exc))
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
