import pandas as pd

from src.recommender.candidate_generator import CandidateGenerator
from src.recommender.hybrid_scorer import HybridScorer
from src.recommender.knowledge_base import initialize_knowledge_base
from src.recommender.path_planner import PathPlanner
from src.recommender.risk_rules import generate_weak_labels


def test_xapi_weak_labels_do_not_depend_on_true_class():
    rows = [
        {
            "StudentAbsenceDays": "Above-7",
            "VisITedResources": 10,
            "raisedhands": 12,
            "Discussion": 8,
            "AnnouncementsView": 15,
            "ParentAnsweringSurvey": "No",
            "ParentschoolSatisfaction": "Bad",
            "Class": "L",
        },
        {
            "StudentAbsenceDays": "Above-7",
            "VisITedResources": 10,
            "raisedhands": 12,
            "Discussion": 8,
            "AnnouncementsView": 15,
            "ParentAnsweringSurvey": "No",
            "ParentschoolSatisfaction": "Bad",
            "Class": "H",
        },
    ]
    labels = generate_weak_labels(pd.DataFrame(rows), "xapi")
    assert labels[0].tolist() == labels[1].tolist()


def test_candidate_generator_prediction_aware_thresholds(tmp_path):
    catalog_df, _ = initialize_knowledge_base(tmp_path)
    generator = CandidateGenerator(catalog_df)
    risks = {
        "R1_LOW_PRIOR_PERFORMANCE": 0.25,
        "R2_DECLINING_TREND": 0.10,
        "R3_ATTENDANCE_RISK": 0.20,
        "R4_LOW_ENGAGEMENT": 0.20,
        "R5_INSUFFICIENT_STUDY_TIME": 0.10,
        "R6_HIGH_FAILURE_PROBABILITY": 0.24,
    }

    low_candidates = generator.generate_candidates(risks, 0, class_probabilities=[0.80, 0.15, 0.05])
    high_candidates = generator.generate_candidates(risks, 2, class_probabilities=[0.05, 0.15, 0.80])

    assert not low_candidates.empty
    assert not high_candidates.empty
    assert len(low_candidates) >= len(high_candidates)
    assert "advanced_seminar" in set(high_candidates["item_id"])


def test_xapi_no_risk_medium_uses_general_candidates(tmp_path):
    catalog_df, _ = initialize_knowledge_base(tmp_path)
    generator = CandidateGenerator(catalog_df)
    risks = {
        "R3_ATTENDANCE_RISK": 0.0,
        "R4_LOW_ENGAGEMENT": 0.0,
        "R6_HIGH_FAILURE_PROBABILITY": 0.0,
    }
    candidates = generator.generate_candidates(
        risks,
        1,
        class_probabilities=[0.05, 0.70, 0.25],
        dataset_kind="xapi",
    )
    item_ids = set(candidates["item_id"])

    assert {"weekly_progress_review", "standard_practice_plan", "maintain_lms_engagement"}.issubset(item_ids)
    assert "attendance_monitoring" not in item_ids
    assert "counselor_meeting" not in item_ids


def test_student_academic_risk_prioritizes_remediation(tmp_path):
    catalog_df, mapping_df = initialize_knowledge_base(tmp_path)
    generator = CandidateGenerator(catalog_df)
    risks = {
        "R1_LOW_PRIOR_PERFORMANCE": 1.0,
        "R2_DECLINING_TREND": 1.0,
        "R3_ATTENDANCE_RISK": 0.0,
        "R4_LOW_ENGAGEMENT": 0.0,
        "R5_INSUFFICIENT_STUDY_TIME": 0.0,
        "R6_HIGH_FAILURE_PROBABILITY": 0.2,
    }
    candidates = generator.generate_candidates(
        risks,
        0,
        class_probabilities=[0.75, 0.25, 0.0],
        dataset_kind="student",
    )
    scorer = HybridScorer(catalog_df, mapping_df)
    recs = scorer.score_student(
        student_features={"studytime": 2.0, "absences": 0.0},
        diagnosed_risks=risks,
        class_probabilities=[0.75, 0.25, 0.0],
        predicted_class=0,
        dataset_kind="student",
        candidates_df=candidates,
    )
    top3 = {rec["item_id"] for rec in recs[:3]}

    assert len(top3.intersection({"extra_exercises", "peer_tutoring", "remedial_class", "academic_coaching"})) >= 2
    assert "resource_checklist" not in top3


def test_hybrid_scorer_returns_sorted_results_with_prediction_context(tmp_path):
    catalog_df, mapping_df = initialize_knowledge_base(tmp_path)
    scorer = HybridScorer(catalog_df, mapping_df)
    recs = scorer.score_student(
        student_features={"studytime": 2.0, "absences": 1.0},
        diagnosed_risks={
            "R1_LOW_PRIOR_PERFORMANCE": 0.2,
            "R2_DECLINING_TREND": 0.1,
            "R3_ATTENDANCE_RISK": 0.7,
            "R4_LOW_ENGAGEMENT": 0.4,
            "R5_INSUFFICIENT_STUDY_TIME": 0.2,
            "R6_HIGH_FAILURE_PROBABILITY": 0.6,
        },
        class_probabilities=[0.70, 0.20, 0.10],
        predicted_class=0,
        dataset_kind="student",
    )

    scores = [rec["score"] for rec in recs]
    assert all(scores[i] >= scores[i + 1] for i in range(len(scores) - 1))
    assert all("prediction_context" in rec for rec in recs)


def test_path_planner_has_required_four_week_structure():
    planner = PathPlanner()
    path = planner.generate_path(
        scored_interventions=[
            {
                "item_id": "attendance_monitoring",
                "intervention_name": "Daily Attendance Monitoring",
                "description": "Weekly attendance check-ins.",
                "recommended_phase": "Stabilize",
            }
        ],
        predicted_class=0,
        diagnosed_risks={"R3_ATTENDANCE_RISK": 0.8, "R4_LOW_ENGAGEMENT": 0.4},
    )

    assert path["risk_band"] == "High"
    assert "plan_intensity" in path
    assert "top_risks" in path
    assert "max_risk_score" in path
    assert list(path["weeks"].keys()) == ["Week 1", "Week 2", "Week 3", "Week 4"]
