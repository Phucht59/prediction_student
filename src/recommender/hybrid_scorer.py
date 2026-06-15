import pandas as pd
import numpy as np
from typing import Any
from src.recommender.explanation import generate_friendly_explanation

class HybridScorer:
    """
    Scores interventions based on multi-criteria student profiles:
    - risk_match (0.3)
    - performance_need (0.2)
    - difficulty_fit (0.15)
    - time_fit (0.15)
    - prerequisite_fit (0.1)
    - expected_effect (0.1)
    """
    def __init__(self, catalog_df: pd.DataFrame, mapping_df: pd.DataFrame):
        self.catalog_df = catalog_df
        self.mapping_df = mapping_df
        
    def score_student(
        self,
        student_features: dict[str, Any],
        diagnosed_risks: dict[str, float],
        class_probabilities: list[float],
        predicted_class: int,
        dataset_kind: str,
        candidates_df: pd.DataFrame = None
    ) -> list[dict[str, Any]]:
        # 1. Estimate student study capacity
        if "student" in dataset_kind.lower():
            studytime = float(student_features.get("studytime", 1.0))
            if studytime == 1.0:
                capacity = 2.0
            elif studytime == 2.0:
                capacity = 5.0
            elif studytime == 3.0:
                capacity = 8.0
            else:
                capacity = 12.0
            
            absences = float(student_features.get("absences", 0.0))
            absences_penalty = absences * 0.2
        else:
            visited = float(student_features.get("VisITedResources", 50.0))
            capacity = 2.0 + (visited / 100.0) * 8.0
            
            absences_val = str(student_features.get("StudentAbsenceDays", "")).strip().lower()
            absences_penalty = 2.0 if absences_val == "above-7" else 0.0
            
        adjusted_capacity = max(1.0, capacity - absences_penalty)
        student_level = int(predicted_class) # 0: Low, 1: Medium, 2: High
        
        target_df = self.catalog_df if candidates_df is None else candidates_df
        
        results = []
        for _, row in target_df.iterrows():
            item_id = row["item_id"]
            name = row["intervention_name"]
            desc = row["description"]
            difficulty = int(row["difficulty_level"])
            hours = float(row["estimated_hours_per_week"])
            phase = row["recommended_phase"]
            effect = float(row["expected_effect"])
            prereq = int(row["prerequisite_level"])
            
            # 1. Risk match (0.3)
            target_risks_str = str(row["target_risks"])
            if pd.isna(row["target_risks"]):
                target_risks_str = ""
            target_risks = [r.strip() for r in target_risks_str.split(",") if r.strip()]
            
            if target_risks:
                # Max of diagnosed risk probabilities that this intervention targets
                risk_match = max([diagnosed_risks.get(r, 0.0) for r in target_risks])
            else:
                # If no target risks (e.g. advanced seminar), it matches high performance
                risk_match = 1.0 if student_level == 2 else 0.0
                
            # 2. Performance need (0.2)
            p_low, p_med, p_high = class_probabilities
            if difficulty >= 3 and item_id == "advanced_seminar":
                perf_need = p_high
            elif difficulty >= 2:
                perf_need = p_low * 1.0 + p_med * 0.5
            else:
                perf_need = p_low * 0.8 + p_med * 0.5 + p_high * 0.2
                
            # 3. Difficulty fit (0.15)
            # student_level is in [0, 1, 2], difficulty is in [1, 2, 3]
            # Map difficulty to level: difficulty - 1
            intervention_level = difficulty - 1
            diff_fit = 1.0 - abs(student_level - intervention_level) / 2.0
            
            # 4. Time fit (0.15)
            if hours <= adjusted_capacity:
                time_fit = 1.0
            else:
                time_fit = max(0.0, 1.0 - (hours - adjusted_capacity) / 5.0)
                
            # 5. Prerequisite fit (0.1)
            prereq_fit = 1.0 if student_level >= prereq else 0.0
            
            # 6. Expected effect (0.1)
            expected_effect_fit = effect
            
            # Combine scores using the specified weights
            total_score = (
                0.30 * risk_match +
                0.20 * perf_need +
                0.15 * diff_fit +
                0.15 * time_fit +
                0.10 * prereq_fit +
                0.10 * expected_effect_fit
            )
            
            rec_dict = {
                "item_id": item_id,
                "intervention_name": name,
                "description": desc,
                "score": float(total_score),
                "risk_match": float(risk_match),
                "performance_need": float(perf_need),
                "difficulty_fit": float(diff_fit),
                "time_fit": float(time_fit),
                "prerequisite_fit": float(prereq_fit),
                "expected_effect": float(effect),
                "recommended_phase": phase,
                "estimated_hours_per_week": hours,
            }
            
            # Generate friendly explanation and append breakdown
            friendly_exp = generate_friendly_explanation(rec_dict, diagnosed_risks, dataset_kind)
            breakdown = f" (Risk Match: {risk_match:.2f}, Perf Need: {perf_need:.2f}, Diff Fit: {diff_fit:.2f}, Time Fit: {time_fit:.2f}, Prereq Fit: {prereq_fit:.2f}, Effect: {effect:.2f})"
            rec_dict["explanation"] = friendly_exp + breakdown
            
            results.append(rec_dict)
            
        # Sort by overall score descending
        results = sorted(results, key=lambda x: x["score"], reverse=True)
        return results
