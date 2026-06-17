import pytest
import numpy as np
import pandas as pd
import torch
import tempfile
from pathlib import Path

from src.recommender.risk_rules import generate_weak_labels
from src.recommender.risk_head import RiskDiagnosisHead, train_risk_head
from src.recommender.knowledge_base import initialize_knowledge_base, load_knowledge_base
from src.recommender.hybrid_scorer import HybridScorer
from src.recommender.path_planner import PathPlanner
from src.evaluation.recommender_eval import evaluate_risk_diagnosis, evaluate_ranking, evaluate_path_quality

def test_rules_generate_weak_labels_student():
    # Test student logic mapping
    df = pd.DataFrame([
        # R1: R1_LOW_PRIOR_PERFORMANCE (G1 < 10)
        {"absences": 2, "studytime": 2.0, "failures": 0, "G1": 8, "G2": 9, "G3": 12, "goout": 2, "freetime": 2, "activities": "yes"},
        # R2: R2_DECLINING_TREND (G2 < G1)
        {"absences": 2, "studytime": 2.0, "failures": 0, "G1": 12, "G2": 10, "G3": 12, "goout": 2, "freetime": 2, "activities": "yes"},
        # R3: R3_ATTENDANCE_RISK (absences >= 10)
        {"absences": 12, "studytime": 2.0, "failures": 0, "G1": 12, "G2": 12, "G3": 12, "goout": 2, "freetime": 2, "activities": "yes"},
        # R4: R4_LOW_ENGAGEMENT (goout >= 4)
        {"absences": 2, "studytime": 2.0, "failures": 0, "G1": 12, "G2": 12, "G3": 12, "goout": 4, "freetime": 2, "activities": "yes"},
        # R5: R5_INSUFFICIENT_STUDY_TIME (studytime <= 1)
        {"absences": 2, "studytime": 1.0, "failures": 0, "G1": 12, "G2": 12, "G3": 12, "goout": 2, "freetime": 2, "activities": "yes"},
        # R6: R6_HIGH_FAILURE_PROBABILITY from observable failure/grade trajectory, not G3
        {"absences": 2, "studytime": 2.0, "failures": 0, "G1": 12, "G2": 8, "G3": 14, "goout": 2, "freetime": 2, "activities": "yes"},
    ])
    
    labels = generate_weak_labels(df, "student")
    assert labels.shape == (6, 6)
    # R1 target for first row
    assert labels[0, 0] == 1.0
    # R2 target for second row
    assert labels[1, 1] == 1.0
    # R3 target for third row
    assert labels[2, 2] == 1.0
    # R4 target for fourth row
    assert labels[3, 3] == 1.0
    # R5 target for fifth row
    assert labels[4, 4] == 1.0
    # R6 target for sixth row
    assert labels[5, 5] == 1.0

def test_rules_generate_weak_labels_xapi():
    # Test xapi logic mapping
    df = pd.DataFrame([
        # Row 0 -> R3_ATTENDANCE_RISK (StudentAbsenceDays == 'Above-7')
        {"StudentAbsenceDays": "Above-7", "VisITedResources": 50, "raisedhands": 50, "Discussion": 50, "AnnouncementsView": 50, "Class": "H"},
        # Row 1 -> R4_LOW_ENGAGEMENT (VisITedResources < 30)
        {"StudentAbsenceDays": "Under-7", "VisITedResources": 20, "raisedhands": 50, "Discussion": 50, "AnnouncementsView": 50, "Class": "H"},
        # Row 2 -> R6_HIGH_FAILURE_PROBABILITY from observable absence/engagement/support signals, not Class
        {"StudentAbsenceDays": "Above-7", "VisITedResources": 10, "raisedhands": 10, "Discussion": 10, "AnnouncementsView": 10, "ParentAnsweringSurvey": "No", "ParentschoolSatisfaction": "Bad", "Class": "H"},
    ])
    
    labels = generate_weak_labels(df, "xapi")
    assert labels.shape == (3, 3)
    assert labels[0, 0] == 1.0
    assert labels[1, 1] == 1.0
    assert labels[2, 2] == 1.0

def test_risk_head_and_training():
    features = np.random.randn(20, 10)
    class_probs = np.random.dirichlet(np.ones(3), size=20)
    targets = np.random.randint(0, 2, size=(20, 6)).astype(np.float32)
    
    # Train the MLP risk head model
    risk_model = train_risk_head(
        features=features,
        class_probs=class_probs,
        targets=targets,
        epochs=5,
        lr=0.01,
        device="cpu"
    )
    
    # Verify outputs
    probs = risk_model.predict_proba(features, class_probs, device="cpu")
    assert probs.shape == (20, 6)
    assert np.all(probs >= 0.0) and np.all(probs <= 1.0)

def test_knowledge_base_loading():
    with tempfile.TemporaryDirectory() as temp_dir:
        catalog_df, mapping_df = initialize_knowledge_base(temp_dir)
        
        # Verify schema
        assert "item_id" in catalog_df.columns
        assert "intervention_name" in catalog_df.columns
        assert "target_risks" in catalog_df.columns
        assert "applicable_kind" in catalog_df.columns
        assert "difficulty_level" in catalog_df.columns
        assert len(catalog_df) >= 12
        
        # Verify mapping file
        assert "risk_code" in mapping_df.columns
        assert "item_id" in mapping_df.columns
        
        # Re-load
        cat2, map2 = load_knowledge_base(temp_dir)
        assert len(cat2) == len(catalog_df)
        assert len(map2) == len(mapping_df)

def test_hybrid_scorer_and_path_planner():
    with tempfile.TemporaryDirectory() as temp_dir:
        catalog_df, mapping_df = initialize_knowledge_base(temp_dir)
        scorer = HybridScorer(catalog_df, mapping_df)
        
        student_features = {"studytime": 2.0, "absences": 2.0}
        diagnosed_risks = {
            "R1_LOW_PRIOR_PERFORMANCE": 0.1,
            "R2_DECLINING_TREND": 0.2,
            "R3_ATTENDANCE_RISK": 0.8,
            "R4_LOW_ENGAGEMENT": 0.3,
            "R5_INSUFFICIENT_STUDY_TIME": 0.1,
            "R6_HIGH_FAILURE_PROBABILITY": 0.4
        }
        class_probs = [0.6, 0.3, 0.1] # High low probability, so student is struggling
        pred_class = 0
        
        recs = scorer.score_student(student_features, diagnosed_risks, class_probs, pred_class, "student")
        
        assert len(recs) == len(catalog_df)
        # Verify recommendation order is based on score descending
        scores = [r["score"] for r in recs]
        assert all(scores[i] >= scores[i+1] for i in range(len(scores)-1))
        
        # Test PathPlanner
        planner = PathPlanner()
        path = planner.generate_path(recs, pred_class, diagnosed_risks)
        
        assert path["risk_band"] == "High"
        assert path["predicted_class"] == 0
        assert "weeks" in path
        for week_name in ["Week 1", "Week 2", "Week 3", "Week 4"]:
            assert week_name in path["weeks"]
            assert "theme" in path["weeks"][week_name]
            assert "objective" in path["weeks"][week_name]
            assert "recommended_actions" in path["weeks"][week_name]

def test_evaluation_metrics():
    # Mocking inputs to evaluate_risk_diagnosis, evaluate_ranking, evaluate_path_quality
    y_true = np.array([[1, 0, 1], [0, 1, 0]])
    y_pred = np.array([[0.8, 0.2, 0.9], [0.1, 0.7, 0.3]])
    
    risk_metrics = evaluate_risk_diagnosis(y_true, y_pred)
    assert "f1_micro" in risk_metrics
    assert "hamming_loss" in risk_metrics
    assert risk_metrics["f1_micro"] > 0
    
    # Mock for ranking evaluation
    recs_list = [
        [{"item_id": "attendance_monitoring"}, {"item_id": "counselor_meeting"}, {"item_id": "time_planning"}],
        [{"item_id": "extra_exercises"}, {"item_id": "study_group"}, {"item_id": "peer_tutoring"}]
    ]
    actual_risks_list = [
        ["R3_ATTENDANCE_RISK"],
        ["R1_LOW_PRIOR_PERFORMANCE", "R4_LOW_ENGAGEMENT"]
    ]
    
    with tempfile.TemporaryDirectory() as temp_dir:
        catalog_df, mapping_df = initialize_knowledge_base(temp_dir)
        ranking_metrics = evaluate_ranking(recs_list, actual_risks_list, catalog_df, k=3)
        assert "precision_at_3" in ranking_metrics
        assert "recall_at_3" in ranking_metrics
        assert "ndcg_at_3" in ranking_metrics
        
        # Mock paths for quality evaluation
        paths_list = [
            {
                "predicted_class": 0,
                "weeks": {
                    "Week 1": {"item_ids": ["attendance_monitoring", "counselor_meeting"]},
                    "Week 2": {"item_ids": ["extra_exercises"]},
                    "Week 3": {"item_ids": ["study_group"]},
                    "Week 4": {"item_ids": []}
                }
            },
            {
                "predicted_class": 1,
                "weeks": {
                    "Week 1": {"item_ids": ["time_planning"]},
                    "Week 2": {"item_ids": ["peer_tutoring"]},
                    "Week 3": {"item_ids": ["study_group"]},
                    "Week 4": {"item_ids": []}
                }
            }
        ]
        path_metrics = evaluate_path_quality(paths_list, actual_risks_list, catalog_df)
        assert "risk_coverage_rate" in path_metrics
        assert "workload_balance_std" in path_metrics
        assert "difficulty_progression_rate" in path_metrics
        assert "prerequisite_violation_rate" in path_metrics
