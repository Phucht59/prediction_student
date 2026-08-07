from pathlib import Path

import pandas as pd

from src.recommend_hybrid.explainable_v2.query_evidence import (
    build_query_level_evidence,
    expand_action_candidates,
)


def _write_raw(root: Path) -> None:
    raw = root / "data/raw"
    raw.mkdir(parents=True)

    pd.DataFrame(
        [
            {
                "code_module": "AAA",
                "code_presentation": "2013J",
                "id_assessment": 1,
                "assessment_type": "TMA",
                "date": 10,
                "weight": 10,
            },
            {
                "code_module": "AAA",
                "code_presentation": "2013J",
                "id_assessment": 2,
                "assessment_type": "TMA",
                "date": 25,
                "weight": 10,
            },
        ]
    ).to_csv(
        raw / "assessments.csv",
        index=False,
    )

    pd.DataFrame(
        [
            {
                "id_assessment": 1,
                "id_student": 100,
                "date_submitted": 9,
                "is_banked": 0,
                "score": 80,
            }
        ]
    ).to_csv(
        raw / "studentAssessment.csv",
        index=False,
    )

    pd.DataFrame(
        [
            {
                "id_site": 10,
                "code_module": "AAA",
                "code_presentation": "2013J",
                "activity_type": "oucontent",
                "week_from": 0,
                "week_to": 10,
            },
            {
                "id_site": 11,
                "code_module": "AAA",
                "code_presentation": "2013J",
                "activity_type": "quiz",
                "week_from": 1,
                "week_to": 10,
            },
        ]
    ).to_csv(
        raw / "vle.csv",
        index=False,
    )

    pd.DataFrame(
        [
            {
                "code_module": "AAA",
                "code_presentation": "2013J",
                "id_student": 100,
                "id_site": 10,
                "date": 15,
                "sum_click": 2,
            }
        ]
    ).to_csv(
        raw / "studentVle.csv",
        index=False,
    )


def _learner() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "query_id": (
                    "100::AAA::2013J::EARLY_20"
                ),
                "student_key": "100",
                "course_key": "AAA::2013J",
                "code_module": "AAA",
                "code_presentation": "2013J",
                "outer_fold": 0,
                "stage": "EARLY_20",
                "cutoff_day": 20,
                "risk_probability": 0.8,
                "hybrid_uncertainty": 0.2,
                "seed_disagreement": 0.01,
                "course_progress": 0.2,
                "active_day_rate": 0.3,
                "regularity_score": 0.4,
                "content_coverage": 0.5,
                "quiz_activity": 0.1,
            }
        ]
    )


def test_query_evidence_is_cutoff_safe_and_action_independent(
    tmp_path,
):
    _write_raw(tmp_path)
    evidence = build_query_level_evidence(
        _learner(),
        root=tmp_path,
    )
    row = evidence.iloc[0]

    assert row["assessments_due"] == 1
    assert row["missing_assessment_count"] == 0
    assert row["due_soon_count"] == 1
    assert row["completion_rate"] == 1.0
    assert row["inactivity_streak"] == 4
    assert bool(row["quiz_available"]) is True
    assert bool(row["vle_available"]) is True
    assert bool(
        row["study_material_available"]
    ) is True

    expanded = expand_action_candidates(
        evidence
    )
    assert len(expanded) == 5
    for field in [
        "inactivity_streak",
        "assessments_due",
        "missing_assessment_count",
        "quiz_available",
    ]:
        assert (
            expanded[field]
            .nunique(dropna=False)
            == 1
        )


def test_query_evidence_excludes_post_cutoff_submission(
    tmp_path,
):
    _write_raw(tmp_path)
    path = (
        tmp_path
        / "data/raw/studentAssessment.csv"
    )
    frame = pd.read_csv(path)
    frame.loc[0, "date_submitted"] = 21
    frame.to_csv(path, index=False)

    evidence = build_query_level_evidence(
        _learner(),
        root=tmp_path,
    )
    row = evidence.iloc[0]
    assert row["assessments_due"] == 1
    assert row["missing_assessment_count"] == 1
    assert row["completion_rate"] == 0.0
