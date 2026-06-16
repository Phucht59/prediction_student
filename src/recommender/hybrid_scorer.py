import pandas as pd
import numpy as np
from typing import Any
from src.recommender.explanation import generate_friendly_explanation


class HybridScorer:
    """
    Scores interventions based on prediction-aware multi-criteria evidence.

    The recommender is intentionally downstream of the prediction model:
    predicted class probabilities control intervention urgency, while diagnosed
    risks and student profile signals choose the concrete support actions.
    """

    def __init__(self, catalog_df: pd.DataFrame, mapping_df: pd.DataFrame):
        self.catalog_df = catalog_df
        self.mapping_df = mapping_df

    @staticmethod
    def _adaptive_weights(predicted_class: int, p_low: float, p_high: float, max_risk: float) -> dict[str, float]:
        """Return scoring weights adapted to the predicted academic state."""
        if int(predicted_class) == 0 or p_low >= 0.50 or max_risk >= 0.65:
            return {
                "risk_match": 0.36,
                "performance_need": 0.24,
                "difficulty_fit": 0.14,
                "time_fit": 0.10,
                "prerequisite_fit": 0.06,
                "expected_effect": 0.10,
            }
        if int(predicted_class) == 2 and p_high >= 0.55 and max_risk < 0.45:
            return {
                "risk_match": 0.18,
                "performance_need": 0.12,
                "difficulty_fit": 0.24,
                "time_fit": 0.12,
                "prerequisite_fit": 0.16,
                "expected_effect": 0.18,
            }
        return {
            "risk_match": 0.30,
            "performance_need": 0.20,
            "difficulty_fit": 0.18,
            "time_fit": 0.12,
            "prerequisite_fit": 0.10,
            "expected_effect": 0.10,
        }

    @staticmethod
    def _difficulty_fit(student_level: int, difficulty: int, target_risks: list[str], max_risk: float) -> float:
        """Fit difficulty to performance level without blocking remedial bootcamps."""
        high_failure_item = any(
            risk in {"R1_LOW_PRIOR_PERFORMANCE", "R6_HIGH_FAILURE_PROBABILITY"}
            for risk in target_risks
        )
        if student_level == 0:
            if difficulty == 1:
                return 1.0
            if difficulty == 2:
                return 0.88
            return 0.68 if high_failure_item or max_risk >= 0.70 else 0.35
        if student_level == 1:
            return {1: 0.75, 2: 1.0, 3: 0.72}.get(difficulty, 0.5)
        # High performers should get enrichment unless strong risks exist.
        if max_risk >= 0.55 and high_failure_item:
            return {1: 0.85, 2: 0.90, 3: 0.70}.get(difficulty, 0.5)
        return {1: 0.50, 2: 0.82, 3: 1.0}.get(difficulty, 0.5)

    @staticmethod
    def _estimate_capacity(student_features: dict[str, Any], dataset_kind: str) -> tuple[float, float]:
        """Estimate weekly study capacity and absence/support penalty."""
        if "student" in dataset_kind.lower():
            studytime = float(student_features.get("studytime", 1.0))
            capacity = {1.0: 2.0, 2.0: 5.0, 3.0: 8.0}.get(studytime, 12.0)
            absences = float(student_features.get("absences", 0.0))
            absences_penalty = absences * 0.2
            return capacity, absences_penalty

        visited = float(student_features.get("VisITedResources", 50.0))
        raised = float(student_features.get("raisedhands", 50.0))
        discussion = float(student_features.get("Discussion", 50.0))
        announcements = float(student_features.get("AnnouncementsView", 50.0))
        engagement_avg = np.mean([visited, raised, discussion, announcements])
        capacity = 2.0 + (engagement_avg / 100.0) * 8.0

        absences_val = str(student_features.get("StudentAbsenceDays", "")).strip().lower()
        parent_answer = str(student_features.get("ParentAnsweringSurvey", "")).strip().lower()
        school_satisfaction = str(student_features.get("ParentschoolSatisfaction", "")).strip().lower()
        absences_penalty = 2.0 if absences_val == "above-7" else 0.0
        support_penalty = 0.75 if parent_answer == "no" or school_satisfaction == "bad" else 0.0
        return capacity, absences_penalty + support_penalty

    def score_student(
        self,
        student_features: dict[str, Any],
        diagnosed_risks: dict[str, float],
        class_probabilities: list[float],
        predicted_class: int,
        dataset_kind: str,
        candidates_df: pd.DataFrame = None,
    ) -> list[dict[str, Any]]:
        capacity, penalty = self._estimate_capacity(student_features, dataset_kind)
        adjusted_capacity = max(1.0, capacity - penalty)
        student_level = int(predicted_class)  # 0: Low, 1: Medium, 2: High
        p_low, p_med, p_high = [float(x) for x in class_probabilities[:3]]
        max_risk = max([float(v) for v in diagnosed_risks.values()], default=0.0)
        weights = self._adaptive_weights(student_level, p_low, p_high, max_risk)

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

            target_risks_str = str(row["target_risks"])
            if pd.isna(row["target_risks"]):
                target_risks_str = ""
            target_risks = [r.strip() for r in target_risks_str.split(",") if r.strip()]

            if target_risks:
                matched_scores = [diagnosed_risks.get(r, 0.0) for r in target_risks]
                risk_match = max(matched_scores)
                risk_coverage_bonus = min(0.15, 0.05 * sum(float(v) >= 0.45 for v in matched_scores))
                risk_match = min(1.0, risk_match + risk_coverage_bonus)
            else:
                risk_match = 1.0 if student_level == 2 and max_risk < 0.45 else 0.0

            # Performance need is driven by prediction probabilities. Low/uncertain
            # students receive remedial support; high-confidence High students receive enrichment.
            if not target_risks and difficulty >= 3:
                perf_need = p_high
            elif any(r in {"R1_LOW_PRIOR_PERFORMANCE", "R6_HIGH_FAILURE_PROBABILITY"} for r in target_risks):
                perf_need = min(1.0, p_low + 0.40 * p_med + 0.20 * max_risk)
            elif any(r in {"R3_ATTENDANCE_RISK", "R4_LOW_ENGAGEMENT", "R5_INSUFFICIENT_STUDY_TIME"} for r in target_risks):
                perf_need = min(1.0, 0.75 * p_low + 0.55 * p_med + 0.25 * max_risk)
            else:
                perf_need = p_low * 0.60 + p_med * 0.45 + p_high * 0.25

            diff_fit = self._difficulty_fit(student_level, difficulty, target_risks, max_risk)

            if hours <= adjusted_capacity:
                time_fit = 1.0
            else:
                time_fit = max(0.0, 1.0 - (hours - adjusted_capacity) / 5.0)

            # If predicted Low, allow prerequisite level 1 interventions because they
            # are remedial. Advanced prerequisite violations remain penalized.
            if student_level >= prereq:
                prereq_fit = 1.0
            elif student_level == 0 and prereq <= 1 and any(
                r in {"R1_LOW_PRIOR_PERFORMANCE", "R6_HIGH_FAILURE_PROBABILITY"} for r in target_risks
            ):
                prereq_fit = 0.85
            else:
                prereq_fit = 0.0

            expected_effect_fit = effect
            total_score = (
                weights["risk_match"] * risk_match
                + weights["performance_need"] * perf_need
                + weights["difficulty_fit"] * diff_fit
                + weights["time_fit"] * time_fit
                + weights["prerequisite_fit"] * prereq_fit
                + weights["expected_effect"] * expected_effect_fit
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
                "prediction_context": {
                    "predicted_class": int(predicted_class),
                    "p_low": round(p_low, 4),
                    "p_medium": round(p_med, 4),
                    "p_high": round(p_high, 4),
                    "max_diagnosed_risk": round(max_risk, 4),
                    "adjusted_capacity_hours": round(adjusted_capacity, 2),
                },
            }

            friendly_exp = generate_friendly_explanation(rec_dict, diagnosed_risks, dataset_kind)
            breakdown = (
                f" (Risk Match: {risk_match:.2f}, Perf Need: {perf_need:.2f}, "
                f"Diff Fit: {diff_fit:.2f}, Time Fit: {time_fit:.2f}, "
                f"Prereq Fit: {prereq_fit:.2f}, Effect: {effect:.2f}, "
                f"pLow: {p_low:.2f}, MaxRisk: {max_risk:.2f})"
            )
            rec_dict["explanation"] = friendly_exp + breakdown
            results.append(rec_dict)

        results = sorted(results, key=lambda x: x["score"], reverse=True)
        return results
