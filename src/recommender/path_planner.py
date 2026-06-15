from typing import Any

class PathPlanner:
    """
    Groups and schedules recommended interventions into a structured 4-week learning path.
    Themes:
    - Week 1: Stabilize
    - Week 2: Practice
    - Week 3: Reinforce
    - Week 4: Evaluate & Adjust
    """
    def __init__(self):
        pass
        
    def generate_path(
        self,
        scored_interventions: list[dict[str, Any]],
        predicted_class: int,
        diagnosed_risks: dict[str, float]
    ) -> dict[str, Any]:
        # Determine risk band
        max_risk = max(diagnosed_risks.values()) if diagnosed_risks else 0.0
        if predicted_class == 0 or max_risk >= 0.5:
            risk_band = "High"
        elif predicted_class == 1 or max_risk >= 0.3:
            risk_band = "Moderate"
        else:
            risk_band = "Stable"
            
        # Group interventions by phase
        phase_map: dict[str, list[dict[str, Any]]] = {
            "Stabilize": [],
            "Practice": [],
            "Reinforce": [],
            "Evaluate & Adjust": []
        }
        
        # Consider top-scoring interventions (e.g., score >= 0.35)
        # We always keep at least the top 5 to distribute across weeks if possible
        top_interventions = scored_interventions[:6]
        for item in top_interventions:
            phase = item["recommended_phase"]
            if phase in phase_map:
                phase_map[phase].append(item)
                
        # Construct the 4-week plan
        weeks = {}
        
        # Week 1: Stabilize
        w1_actions = phase_map["Stabilize"]
        if w1_actions:
            w1_rec_actions = [f"{a['intervention_name']}: {a['description']}" for a in w1_actions]
            w1_objective = "Establish basic academic stability and resolve immediate attendance or support barriers."
            w1_outcome = "Regular class attendance established and a structured weekly study schedule created."
            w1_explanation = f"Addressed high priority risks ({', '.join([a['item_id'] for a in w1_actions])}) to build a stable learning foundation."
            w1_ids = [a["item_id"] for a in w1_actions]
        else:
            w1_rec_actions = ["Maintain current attendance and standard study schedules. No urgent stabilization required."]
            w1_objective = "Maintain current academic stability and check-in with advisor."
            w1_outcome = "Consistent attendance and stable scheduling maintained."
            w1_explanation = "No critical stabilization issues detected based on current student profile."
            w1_ids = []
            
        weeks["Week 1"] = {
            "theme": "Stabilize",
            "objective": w1_objective,
            "recommended_actions": w1_rec_actions,
            "expected_outcome": w1_outcome,
            "explanation": w1_explanation,
            "item_ids": w1_ids
        }
        
        # Week 2: Practice
        w2_actions = phase_map["Practice"]
        if w2_actions:
            w2_rec_actions = [f"{a['intervention_name']}: {a['description']}" for a in w2_actions]
            w2_objective = "Remediate core knowledge gaps and practice key concepts to catch up."
            w2_outcome = "Completion of initial practice exercises and reduction in concept gaps."
            w2_explanation = f"Prioritizes targeted tasks ({', '.join([a['item_id'] for a in w2_actions])}) to reinforce basic subject mastery."
            w2_ids = [a["item_id"] for a in w2_actions]
        else:
            w2_rec_actions = ["Complete standard homework assignments. Optional: review previous exam questions."]
            w2_objective = "Standard practice and concept review."
            w2_outcome = "Consistent completion of weekly homework."
            w2_explanation = "No significant knowledge gaps diagnosed. Continuing with standard practice."
            w2_ids = []
            
        weeks["Week 2"] = {
            "theme": "Practice",
            "objective": w2_objective,
            "recommended_actions": w2_rec_actions,
            "expected_outcome": w2_outcome,
            "explanation": w2_explanation,
            "item_ids": w2_ids
        }
        
        # Week 3: Reinforce
        w3_actions = phase_map["Reinforce"]
        if w3_actions:
            w3_rec_actions = [f"{a['intervention_name']}: {a['description']}" for a in w3_actions]
            w3_objective = "Engage in collaborative study and leverage interactive resources to deepen understanding."
            w3_outcome = "Active participation in peer groups and increased digital platform engagement."
            w3_explanation = f"Uses interactive activities ({', '.join([a['item_id'] for a in w3_actions])}) to sustain motivation and learning speed."
            w3_ids = [a["item_id"] for a in w3_actions]
        else:
            w3_rec_actions = ["Participate in general class discussions. Use LMS resources for regular reading."]
            w3_objective = "General reinforcement and digital resource reading."
            w3_outcome = "Steady engagement with digital course resources."
            w3_explanation = "Student engagement and habits are already at acceptable levels."
            w3_ids = []
            
        weeks["Week 3"] = {
            "theme": "Reinforce",
            "objective": w3_objective,
            "recommended_actions": w3_rec_actions,
            "expected_outcome": w3_outcome,
            "explanation": w3_explanation,
            "item_ids": w3_ids
        }
        
        # Week 4: Evaluate & Adjust
        w4_actions = phase_map["Evaluate & Adjust"]
        w4_ids = []
        if w4_actions and predicted_class == 2:
            # Only recommend advanced seminar for High performing students
            w4_rec_actions = [f"{a['intervention_name']}: {a['description']}" for a in w4_actions]
            w4_objective = "Pursue advanced topics to challenge capacity and expand skills."
            w4_outcome = "Completion of an enrichment topic or advanced challenge."
            w4_explanation = "High performance indicates capability to handle advanced challenges."
            w4_ids = [a["item_id"] for a in w4_actions]
        else:
            w4_rec_actions = ["Review weekly study metrics and grade logs. Plan study targets for the next month."]
            w4_objective = "Self-evaluate progress and adjust study goals for the coming cycle."
            w4_outcome = "Clear understanding of progress and updated self-study goals."
            w4_explanation = "Cycle wrap-up: reflection on achievements and setting goals for the next month."
            
        weeks["Week 4"] = {
            "theme": "Evaluate & Adjust",
            "objective": w4_objective,
            "recommended_actions": w4_rec_actions,
            "expected_outcome": w4_outcome,
            "explanation": w4_explanation,
            "item_ids": w4_ids
        }
        
        return {
            "risk_band": risk_band,
            "predicted_class": int(predicted_class),
            "weeks": weeks
        }
