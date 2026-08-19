"""Join portable Panel A Gemini/Snorkel labels. Never read Panel B reviews."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.recommend_hybrid.v3.contracts import CanonicalAction, RecommendationFeatures, Stage
from src.recommend_hybrid.v3.feasibility import evaluate_action
from src.recommend_hybrid.v3.weak_labels import aggregate, behavioral_vote, feasibility_vote, fit_label_model, gemini_vote

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "artifacts" / "recommend_hybrid" / "v3" / "labels"
DATA = ROOT / "artifacts" / "recommend_hybrid" / "v3" / "data"
PANEL_A = ROOT / "artifacts" / "recommend_hybrid" / "final" / "panel_a_reviews" / "panel_a_external_reviews_frozen.jsonl"
SILVER = ROOT / "artifacts" / "recommend_hybrid" / "final" / "weak_labels" / "probabilistic_relevance_labels.parquet"


def _features_from_row(row: pd.Series) -> RecommendationFeatures:
    def maybe_int(name):
        value = row.get(name)
        return None if value is None or pd.isna(value) else int(value)

    def maybe_float(name):
        value = row.get(name)
        return None if value is None or pd.isna(value) else float(value)

    return RecommendationFeatures(
        student_key=str(row["student_key"]),
        course_key=str(row["course_key"]),
        record_id=str(row["record_id"]),
        stage=Stage(str(row["stage"])),
        cutoff_day=int(row["cutoff_day"]),
        risk_probability=float(row["risk_probability"]),
        predicted_risk=int(row["predicted_risk"]),
        prediction_threshold=float(row["prediction_threshold"]),
        uncertainty=float(row["uncertainty"]),
        course_progress=float(row["course_progress"]),
        missing_assessment_count=maybe_int("missing_assessment_count"),
        due_soon_count=maybe_int("due_soon_count"),
        completion_rate=maybe_float("completion_rate"),
        quiz_available=bool(row["quiz_available"]) if pd.notna(row.get("quiz_available")) else None,
        vle_access_available=bool(row["vle_access_available"]) if pd.notna(row.get("vle_access_available")) else None,
        study_material_available=bool(row["study_material_available"]) if pd.notna(row.get("study_material_available")) else None,
        active_day_rate=maybe_float("active_day_rate"),
        regularity_score=maybe_float("regularity_score"),
        content_coverage=maybe_float("content_coverage"),
        inactivity_streak=maybe_int("inactivity_streak"),
        quiz_activity=maybe_float("quiz_activity"),
    )


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    features = pd.read_parquet(DATA / "learner_stage_features.parquet")
    silver = pd.read_parquet(SILVER)
    gemini = pd.DataFrame(
        [
            {"case_id": json.loads(line)["case_id"], "action_id": json.loads(line)["action_id"], "gemini_score": json.loads(line)["relevance_score"]}
            for line in PANEL_A.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    )
    silver = silver.merge(gemini, on=["case_id", "action_id"], how="left")
    matched = features.merge(
        silver[["query_id", "action_id", "case_id", "gemini_score"]].rename(columns={"gemini_score": "gemini_score"}),
        on="query_id",
        how="inner",
    )
    matched["portability_status"] = "CONDITIONALLY_PORTABLE"
    unmatched_queries = features.loc[~features.query_id.isin(set(silver.query_id))].copy()
    extras = []
    for action in CanonicalAction:
        part = unmatched_queries.copy()
        part["action_id"] = action.value
        part["case_id"] = pd.NA
        part["gemini_score"] = np.nan
        part["portability_status"] = "UNMATCHED"
        extras.append(part)
    rows = pd.concat([matched, *extras], ignore_index=True) if len(extras) else matched
    rows = rows.drop_duplicates(["query_id", "action_id"], keep="first")
    eligible = []
    votes = []
    for _, row in rows.iterrows():
        feats = _features_from_row(row)
        is_eligible = evaluate_action(CanonicalAction(row["action_id"]), feats).eligible
        eligible.append(is_eligible)
        votes.append(
            [
                behavioral_vote(row, row["action_id"]),
                feasibility_vote(is_eligible),
                gemini_vote(row.get("gemini_score")),
            ]
        )
    rows = rows.copy()
    rows["eligible"] = eligible
    rows = rows.drop(columns=[c for c in rows.columns if c in {"expected_relevance", "hard_relevance", "label_status"}])
    model = fit_label_model(np.asarray(votes, dtype=int), seed=2026, epochs=400)
    agg = aggregate(model, np.asarray(votes, dtype=int), min_families=2, source_families=["BEHAVIORAL", "FEASIBILITY", "LLM_EXPERT"])
    out = pd.concat([rows.reset_index(drop=True), agg], axis=1)
    out = out.loc[:, ~out.columns.duplicated()]
    out.to_parquet(OUT / "v3_action_rows.parquet", index=False)
    summary = {
        "panel_b_used": False,
        "feature_queries": int(features.query_id.nunique()),
        "action_rows": int(len(out)),
        "conditionally_portable_rows": int((out.portability_status == "CONDITIONALLY_PORTABLE").sum()),
        "unmatched_rows": int((out.portability_status == "UNMATCHED").sum()),
        "retained": int((out.label_status == "RETAINED").sum()),
        "gemini_non_null": int(out.gemini_score.notna().sum()),
        "matched_queries": int(matched.query_id.nunique()) if len(matched) else 0,
    }
    (OUT / "LABEL_PORTABILITY_SUMMARY.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    (OUT / "WEAK_LABEL_MANIFEST.json").write_text(
        json.dumps({"sources": ["BEHAVIORAL", "FEASIBILITY", "LLM_EXPERT"], "min_families": 2, "panel_b_used": False}, indent=2) + "\n",
        encoding="utf-8",
    )
    print(summary)


if __name__ == "__main__":
    main()
