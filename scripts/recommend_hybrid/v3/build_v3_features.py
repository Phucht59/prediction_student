"""Join C0 OOF risk with cutoff-safe evidence. 100pct excluded."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from src.recommend_hybrid.v3.evidence_builder import build_evidence_table

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "artifacts" / "recommend_hybrid" / "v3" / "data"
RAW = Path(r"C:\hufit\kltn\data\raw")


def main() -> None:
    oof = pd.read_parquet(OUT / "c0_oof_predictions.parquet")
    if (oof.stage_or_endpoint == "100pct").any():
        raise RuntimeError("100pct leaked into C0 OOF table")
    if oof.query_id.duplicated().any():
        raise RuntimeError("duplicate query")
    queries = oof[
        [
            "query_id",
            "record_id",
            "student_key",
            "course_key",
            "id_student",
            "code_module",
            "code_presentation",
            "stage",
            "stage_or_endpoint",
            "cutoff_day",
            "inner_fold",
            "risk_probability",
            "predicted_risk",
            "prediction_threshold",
            "uncertainty",
        ]
    ].copy()
    evidence = build_evidence_table(queries, RAW)
    keep = [
        c
        for c in evidence.columns
        if c not in {"final_result", "target", "date_unregistration", "score"}
    ]
    table = evidence.loc[:, keep]
    if table.query_id.duplicated().any():
        raise RuntimeError("duplicate after evidence join")
    path = OUT / "learner_stage_features.parquet"
    table.to_parquet(path, index=False)
    lineage = pd.DataFrame(
        [
            {"feature": "risk_probability", "family": "prediction", "source": "C0_DOWNSTREAM_OOF", "cutoff_rule": "n/a", "fit_only": False},
            {"feature": "uncertainty", "family": "prediction", "source": "H2(p) of C0 OOF", "cutoff_rule": "n/a", "fit_only": False},
            {"feature": "prediction_threshold", "family": "prediction", "source": "STOP F1/recall/|t-0.5|", "cutoff_rule": "n/a", "fit_only": False},
            {"feature": "missing_assessment_count", "family": "evidence", "source": "assessments+studentAssessment", "cutoff_rule": "due<t and submitted<t", "fit_only": False},
            {"feature": "due_soon_count", "family": "evidence", "source": "assessments", "cutoff_rule": "[cutoff, cutoff+14)", "fit_only": False},
            {"feature": "inactivity_streak", "family": "evidence", "source": "studentVle", "cutoff_rule": "0<=date<cutoff", "fit_only": False},
            {"feature": "active_day_rate", "family": "evidence", "source": "studentVle", "cutoff_rule": "0<=date<cutoff", "fit_only": False},
            {"feature": "regularity_score", "family": "evidence", "source": "studentVle weekly flags", "cutoff_rule": "0<=date<cutoff", "fit_only": False},
            {"feature": "content_coverage", "family": "evidence", "source": "studentVle+vle content types", "cutoff_rule": "0<=date<cutoff", "fit_only": False},
            {"feature": "quiz_activity", "family": "evidence", "source": "studentVle quiz types", "cutoff_rule": "0<=date<cutoff", "fit_only": False},
        ]
    )
    lineage.to_csv(OUT / "FEATURE_LINEAGE.csv", index=False)
    manifest = {
        "row_count": int(len(table)),
        "query_count": int(table.query_id.nunique()),
        "students": int(table.student_key.nunique()),
        "stages": sorted(table.stage.unique().tolist()),
        "hundred_pct_present": False,
        "duplicate_query_count": int(table.query_id.duplicated().sum()),
        "outcome_columns": [c for c in table.columns if c.lower() in {"final_result", "target", "score"}],
        "raw_dir": str(RAW),
    }
    if manifest["outcome_columns"]:
        raise RuntimeError(f"outcome columns leaked: {manifest['outcome_columns']}")
    (OUT / "FEATURE_MANIFEST.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print("WROTE", path, len(table))


if __name__ == "__main__":
    main()
