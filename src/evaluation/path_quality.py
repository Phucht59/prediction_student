import numpy as np
import pandas as pd
from typing import Any

def evaluate_path_quality(
    paths_list: list[dict[str, Any]],
    actual_risks_list: list[list[str]],
    catalog_df: pd.DataFrame
) -> dict[str, float]:
    """
    Calculate:
    - Risk Coverage Rate: % of student actual risks covered by path interventions.
    - Workload Balance: standard deviation of weekly hours.
    - Difficulty Progression: rate of difficulty progression week-over-week.
    - Prerequisite Violation Rate: % of recommended interventions violating student's prerequisite level.
    """
    catalog_dict = catalog_df.set_index("item_id").to_dict("index")
    
    risk_coverages = []
    workload_balances = []
    difficulty_progressions = []
    prereq_violations = []
    
    for path, actual_risks in zip(paths_list, actual_risks_list):
        actual_risks_set = set(actual_risks)
        predicted_class = path["predicted_class"] # 0: Low, 1: Medium, 2: High
        
        # Track weekly metrics
        week_names = ["Week 1", "Week 2", "Week 3", "Week 4"]
        weekly_hours = []
        weekly_difficulties = []
        
        covered_risks = set()
        violations_count = 0
        total_items_count = 0
        
        for w_name in week_names:
            week_data = path["weeks"].get(w_name, {})
            item_ids = week_data.get("item_ids", [])
            
            w_hours = 0.0
            w_diffs = []
            
            for item_id in item_ids:
                if item_id in catalog_dict:
                    item = catalog_dict[item_id]
                    w_hours += float(item["estimated_hours_per_week"])
                    w_diffs.append(int(item["difficulty_level"]))
                    
                    # Accumulate target risks covered
                    target_risks_str = str(item.get("target_risks", ""))
                    item_risks = [r.strip() for r in target_risks_str.split(",") if r.strip()]
                    covered_risks.update(item_risks)
                    
                    # Check prerequisite level
                    prereq = int(item["prerequisite_level"])
                    if predicted_class < prereq:
                        violations_count += 1
                    total_items_count += 1
            
            weekly_hours.append(w_hours)
            weekly_difficulties.append(np.mean(w_diffs) if w_diffs else 1.0)
            
        # 1. Risk Coverage Rate
        if len(actual_risks_set) > 0:
            coverage = len(actual_risks_set & covered_risks) / len(actual_risks_set)
        else:
            coverage = 1.0
        risk_coverages.append(coverage)
        
        # 2. Workload Balance
        workload_balances.append(np.std(weekly_hours))
        
        # 3. Difficulty Progression
        # Rate of progression: (d2>=d1 + d3>=d2 + d4>=d3) / 3.0
        d1, d2, d3, d4 = weekly_difficulties
        prog = (float(d2 >= d1) + float(d3 >= d2) + float(d4 >= d3)) / 3.0
        difficulty_progressions.append(prog)
        
        # 4. Prerequisite Violations
        prereq_violations.append(violations_count / total_items_count if total_items_count > 0 else 0.0)
        
    return {
        "risk_coverage_rate": float(np.mean(risk_coverages)),
        "workload_balance_std": float(np.mean(workload_balances)),
        "difficulty_progression_rate": float(np.mean(difficulty_progressions)),
        "prerequisite_violation_rate": float(np.mean(prereq_violations))
    }
