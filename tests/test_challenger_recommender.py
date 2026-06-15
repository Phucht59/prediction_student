import os
import json
import pandas as pd
import numpy as np
import pytest
from pathlib import Path

from src.recommender.hybrid_scorer import HybridScorer
from src.recommender.path_planner import PathPlanner
from src.recommender.knowledge_base import load_knowledge_base

# Path paths
OUTPUTS_DIR = Path("outputs/recommender/student-por")
METRICS_PATH = OUTPUTS_DIR / "recommender_metrics.json"
PATHS_PATH = OUTPUTS_DIR / "learning_paths.json"
RESULTS_PATH = OUTPUTS_DIR / "recommendation_results.csv"
PREDS_PATH = OUTPUTS_DIR / "risk_predictions.csv"
CATALOG_PATH = OUTPUTS_DIR / "intervention_catalog.csv"
MAPPING_PATH = OUTPUTS_DIR / "risk_intervention_mapping.csv"

def test_outputs_exist_and_schema():
    """1. Verify that the output files exist and match the schema constraints."""
    assert OUTPUTS_DIR.exists()
    
    # Check recommender_metrics.json
    assert METRICS_PATH.exists()
    with open(METRICS_PATH, "r", encoding="utf-8") as f:
        metrics = json.load(f)
    assert "dataset" in metrics
    assert "risk_diagnosis" in metrics
    assert "ranking" in metrics
    assert "path_quality" in metrics
    for key in ["f1_micro", "f1_macro", "precision_micro", "precision_macro", "recall_micro", "recall_macro", "hamming_loss"]:
        assert key in metrics["risk_diagnosis"]
    for key in ["precision_at_3", "recall_at_3", "ndcg_at_3", "coverage_at_3"]:
        assert key in metrics["ranking"]
    for key in ["risk_coverage_rate", "workload_balance_std", "difficulty_progression_rate", "prerequisite_violation_rate"]:
        assert key in metrics["path_quality"]

    # Check learning_paths.json
    assert PATHS_PATH.exists()
    with open(PATHS_PATH, "r", encoding="utf-8") as f:
        paths = json.load(f)
    assert isinstance(paths, list)
    assert len(paths) > 0
    for entry in paths:
        assert "student_index" in entry
        assert "path" in entry
        path_data = entry["path"]
        assert "risk_band" in path_data
        assert "predicted_class" in path_data
        assert "weeks" in path_data
        weeks = path_data["weeks"]
        assert isinstance(weeks, dict)

    # Check recommendation_results.csv
    assert RESULTS_PATH.exists()
    df_results = pd.read_csv(RESULTS_PATH)
    required_cols = {"student_index", "rank", "item_id", "intervention_name", "score", "explanation"}
    assert required_cols.issubset(df_results.columns)

    # Check risk_predictions.csv
    assert PREDS_PATH.exists()
    df_preds = pd.read_csv(PREDS_PATH)
    assert "student_index" in df_preds.columns
    # Must have at least the 6 core risk dimensions
    required_risks = {"R1_LOW_PRIOR_PERFORMANCE", "R2_DECLINING_TREND", "R3_ATTENDANCE_RISK", "R4_LOW_ENGAGEMENT", "R5_INSUFFICIENT_STUDY_TIME", "R6_HIGH_FAILURE_PROBABILITY"}
    assert required_risks.issubset(df_preds.columns)


def test_hybrid_scorer_weights():
    """2. Verify HybridScorer weights computation according to the formula:
    risk_match (0.3), performance_need (0.2), difficulty_fit (0.15), time_fit (0.15), prerequisite_fit (0.1), expected_effect (0.1).
    """
    # Load catalog and mapping
    assert CATALOG_PATH.exists()
    assert MAPPING_PATH.exists()
    catalog_df = pd.read_csv(CATALOG_PATH)
    mapping_df = pd.read_csv(MAPPING_PATH)
    
    scorer = HybridScorer(catalog_df, mapping_df)
    
    # Mock student context
    student_features = {"studytime": 2.0, "absences": 2.0} # Capacity = 5.0 - 0.4 = 4.6
    diagnosed_risks = {
        "R1_LOW_PRIOR_PERFORMANCE": 0.1,
        "R2_DECLINING_TREND": 0.2,
        "R3_ATTENDANCE_RISK": 0.8,
        "R4_LOW_ENGAGEMENT": 0.3,
        "R5_INSUFFICIENT_STUDY_TIME": 0.1,
        "R6_HIGH_FAILURE_PROBABILITY": 0.4
    }
    class_probabilities = [0.6, 0.3, 0.1]
    predicted_class = 0 # Low
    
    scored_items = scorer.score_student(
        student_features=student_features,
        diagnosed_risks=diagnosed_risks,
        class_probabilities=class_probabilities,
        predicted_class=predicted_class,
        dataset_kind="student"
    )
    
    # Perform manual score checks for each item in catalog
    catalog_dict = catalog_df.set_index("item_id").to_dict("index")
    
    # Let's verify each scored item score
    for item in scored_items:
        item_id = item["item_id"]
        cat_item = catalog_dict[item_id]
        
        # 1. Risk match (0.3)
        target_risks_str = str(cat_item["target_risks"])
        target_risks = [r.strip() for r in target_risks_str.split(",") if r.strip()]
        if target_risks:
            expected_risk_match = max([diagnosed_risks.get(r, 0.0) for r in target_risks])
        else:
            expected_risk_match = 1.0 if predicted_class == 2 else 0.0
            
        # 2. Performance need (0.2)
        difficulty = int(cat_item["difficulty_level"])
        p_low, p_med, p_high = class_probabilities
        if difficulty >= 3 and item_id == "advanced_seminar":
            expected_perf_need = p_high
        elif difficulty >= 2:
            expected_perf_need = p_low * 1.0 + p_med * 0.5
        else:
            expected_perf_need = p_low * 0.8 + p_med * 0.5 + p_high * 0.2
            
        # 3. Difficulty fit (0.15)
        intervention_level = difficulty - 1
        expected_diff_fit = 1.0 - abs(predicted_class - intervention_level) / 2.0
        
        # 4. Time fit (0.15)
        # Capacity calculation:
        # studytime = 2.0 -> capacity = 5.0
        # absences = 2.0 -> absences_penalty = 2.0 * 0.2 = 0.4
        # adjusted_capacity = max(1.0, 5.0 - 0.4) = 4.6
        adjusted_capacity = 4.6
        hours = float(cat_item["estimated_hours_per_week"])
        if hours <= adjusted_capacity:
            expected_time_fit = 1.0
        else:
            expected_time_fit = max(0.0, 1.0 - (hours - adjusted_capacity) / 5.0)
            
        # 5. Prerequisite fit (0.1)
        prereq = int(cat_item["prerequisite_level"])
        expected_prereq_fit = 1.0 if predicted_class >= prereq else 0.0
        
        # 6. Expected effect (0.1)
        expected_effect_fit = float(cat_item["expected_effect"])
        
        # Total
        expected_score = (
            0.30 * expected_risk_match +
            0.20 * expected_perf_need +
            0.15 * expected_diff_fit +
            0.15 * expected_time_fit +
            0.10 * expected_prereq_fit +
            0.10 * expected_effect_fit
        )
        
        assert item["score"] == pytest.approx(expected_score, abs=1e-6)
        assert item["risk_match"] == pytest.approx(expected_risk_match, abs=1e-6)
        assert item["performance_need"] == pytest.approx(expected_perf_need, abs=1e-6)
        assert item["difficulty_fit"] == pytest.approx(expected_diff_fit, abs=1e-6)
        assert item["time_fit"] == pytest.approx(expected_time_fit, abs=1e-6)
        assert item["prerequisite_fit"] == pytest.approx(expected_prereq_fit, abs=1e-6)
        assert item["expected_effect"] == pytest.approx(expected_effect_fit, abs=1e-6)


def test_weekly_splits_and_themes():
    """3. Verify learning paths are split exactly into 4 weeks with matching themes:
    Week 1 (Stabilize), Week 2 (Practice), Week 3 (Reinforce), Week 4 (Evaluate & Adjust).
    """
    with open(PATHS_PATH, "r", encoding="utf-8") as f:
        paths = json.load(f)
        
    expected_themes = {
        "Week 1": "Stabilize",
        "Week 2": "Practice",
        "Week 3": "Reinforce",
        "Week 4": "Evaluate & Adjust"
    }
    
    for entry in paths:
        weeks = entry["path"]["weeks"]
        # Exactly 4 weeks
        assert len(weeks) == 4
        assert list(weeks.keys()) == ["Week 1", "Week 2", "Week 3", "Week 4"]
        
        for w_name, theme in expected_themes.items():
            assert weeks[w_name]["theme"] == theme


def test_prerequisite_violations_and_workload():
    """4. Check for prerequisite violations or workload balance in the generated paths."""
    with open(PATHS_PATH, "r", encoding="utf-8") as f:
        paths = json.load(f)
        
    catalog_df = pd.read_csv(CATALOG_PATH)
    catalog_dict = catalog_df.set_index("item_id").to_dict("index")
    
    violations_found = []
    student_workloads = []
    
    for entry in paths:
        student_idx = entry["student_index"]
        predicted_class = entry["path"]["predicted_class"] # 0: Low, 1: Medium, 2: High
        weeks = entry["path"]["weeks"]
        
        student_hours = []
        for w_name in ["Week 1", "Week 2", "Week 3", "Week 4"]:
            week_data = weeks[w_name]
            item_ids = week_data.get("item_ids", [])
            w_hours = 0.0
            
            for item_id in item_ids:
                if item_id in catalog_dict:
                    item = catalog_dict[item_id]
                    # Check prerequisite
                    prereq = int(item["prerequisite_level"])
                    if predicted_class < prereq:
                        violations_found.append({
                            "student_index": student_idx,
                            "predicted_class": predicted_class,
                            "week": w_name,
                            "item_id": item_id,
                            "prerequisite_level": prereq
                        })
                    w_hours += float(item["estimated_hours_per_week"])
            student_hours.append(w_hours)
        student_workloads.append(student_hours)
        
    # Calculate violation rate as mean of student ratios
    student_violation_rates = []
    for entry in paths:
        s_pred_class = entry["path"]["predicted_class"]
        s_weeks = entry["path"]["weeks"]
        s_violations = 0
        s_items = 0
        for w_name in ["Week 1", "Week 2", "Week 3", "Week 4"]:
            for item_id in s_weeks[w_name].get("item_ids", []):
                if item_id in catalog_dict:
                    if s_pred_class < int(catalog_dict[item_id]["prerequisite_level"]):
                        s_violations += 1
                    s_items += 1
        student_violation_rates.append(s_violations / s_items if s_items > 0 else 0.0)
    violation_rate = np.mean(student_violation_rates)
    
    # Summarize findings for report
    num_violations = len(violations_found)
    total_recs = sum(len(w_data.get("item_ids", [])) for entry in paths for w_data in entry["path"]["weeks"].values())
    
    workload_stds = [np.std(hours) for hours in student_workloads]
    mean_workload_std = np.mean(workload_stds)
    
    print(f"\n--- Prerequisite Violation Summary ---")
    print(f"Total recommended items across all students: {total_recs}")
    print(f"Total prerequisite violations: {num_violations}")
    print(f"Calculated Prerequisite Violation Rate: {violation_rate:.6f}")
    if violations_found:
        print("Sample violations:")
        for v in violations_found[:5]:
            print(f"  Student {v['student_index']} (class {v['predicted_class']}) recommended {v['item_id']} (prereq {v['prerequisite_level']}) in {v['week']}")
            
    print(f"\n--- Workload Balance Summary ---")
    print(f"Mean workload std-dev across weeks: {mean_workload_std:.6f} hours/week")
    
    # Assert constraints if applicable or just verify they run and report findings.
    # Note: Prerequisite violations are computed in evaluation report.
    # We want to report the exact rate. We can check that the rate matches what's in metrics.json.
    with open(METRICS_PATH, "r", encoding="utf-8") as f:
        metrics = json.load(f)
    reported_violation_rate = metrics["path_quality"]["prerequisite_violation_rate"]
    reported_workload_std = metrics["path_quality"]["workload_balance_std"]
    
    assert violation_rate == pytest.approx(reported_violation_rate, abs=1e-6)
    assert mean_workload_std == pytest.approx(reported_workload_std, abs=1e-6)
