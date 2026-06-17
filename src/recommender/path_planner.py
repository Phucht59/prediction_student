from typing import Any


class PathPlanner:
    """
    Groups scored interventions into a transparent 4-week learning path.
    """

    PHASES = ("Stabilize", "Practice", "Reinforce", "Evaluate & Adjust")

    def generate_path(
        self,
        scored_interventions: list[dict[str, Any]],
        predicted_class: int,
        diagnosed_risks: dict[str, float],
    ) -> dict[str, Any]:
        max_risk = max(diagnosed_risks.values()) if diagnosed_risks else 0.0
        sorted_risks = sorted(diagnosed_risks.items(), key=lambda item: item[1], reverse=True)
        top_risk_codes = [risk for risk, score in sorted_risks[:3] if score >= 0.30]
        top_risks = [
            {"risk_code": risk, "score": float(score)}
            for risk, score in sorted_risks[:3]
        ]

        if predicted_class == 0 or max_risk >= 0.65:
            risk_band = "High"
            plan_intensity = "intensive"
        elif predicted_class == 1 or max_risk >= 0.35:
            risk_band = "Moderate"
            plan_intensity = "guided"
        else:
            risk_band = "Stable"
            plan_intensity = "maintenance"

        phase_map: dict[str, list[dict[str, Any]]] = {phase: [] for phase in self.PHASES}
        for item in scored_interventions[:8]:
            phase = item.get("recommended_phase", "")
            if phase in phase_map:
                phase_map[phase].append(item)

        weeks: dict[str, dict[str, Any]] = {}
        weeks["Week 1"] = self._build_week(
            phase="Stabilize",
            actions=phase_map["Stabilize"],
            fallback_action="Set a minimum weekly study schedule and complete one advisor or school check-in.",
            objective="Stabilize attendance, support contact and basic study routine.",
            expected_outcome="Immediate barriers are identified and the student has a concrete weekly plan.",
            top_risks=top_risk_codes,
        )
        weeks["Week 2"] = self._build_week(
            phase="Practice",
            actions=phase_map["Practice"],
            fallback_action="Complete standard homework plus one targeted review exercise set.",
            objective="Close the highest-priority knowledge or engagement gap.",
            expected_outcome="The student completes measurable practice tasks and receives feedback.",
            top_risks=top_risk_codes,
        )
        weeks["Week 3"] = self._build_week(
            phase="Reinforce",
            actions=phase_map["Reinforce"],
            fallback_action="Join class discussion or LMS activity at least twice during the week.",
            objective="Reinforce learning through interaction, resources and repetition.",
            expected_outcome="Engagement indicators improve and weak topics are revisited.",
            top_risks=top_risk_codes,
        )

        evaluate_actions = phase_map["Evaluate & Adjust"]
        if predicted_class == 2 and risk_band == "Stable" and evaluate_actions:
            week4_action_texts = [self._format_action(action) for action in evaluate_actions[:2]]
            week4_ids = [action["item_id"] for action in evaluate_actions[:2]]
            week4_explanation = "Stable prediction and low diagnosed risk allow enrichment-oriented follow-up."
        else:
            week4_action_texts = [
                "Review attendance, LMS/resource usage and practice completion; compare with Week 1 baseline.",
                "If risk indicators remain high, continue the strongest Week 2 intervention for another cycle.",
            ]
            week4_ids = []
            week4_explanation = "The final week evaluates whether the intervention reduced the predicted risk signals."

        weeks["Week 4"] = {
            "theme": "Evaluate & Adjust",
            "objective": "Evaluate progress and decide whether to continue, reduce or escalate support.",
            "recommended_actions": week4_action_texts,
            "expected_outcome": "A clear next-cycle decision based on measured progress.",
            "explanation": week4_explanation,
            "item_ids": week4_ids,
        }

        return {
            "risk_band": risk_band,
            "plan_intensity": plan_intensity,
            "predicted_class": int(predicted_class),
            "top_risks": top_risks,
            "max_risk_score": round(float(max_risk), 4),
            "weeks": weeks,
        }

    @staticmethod
    def _format_action(action: dict[str, Any]) -> str:
        return f"{action['intervention_name']}: {action['description']}"

    def _build_week(
        self,
        phase: str,
        actions: list[dict[str, Any]],
        fallback_action: str,
        objective: str,
        expected_outcome: str,
        top_risks: list[str],
    ) -> dict[str, Any]:
        selected_actions = actions[:2]
        if selected_actions:
            recommended_actions = [self._format_action(action) for action in selected_actions]
            item_ids = [action["item_id"] for action in selected_actions]
            explanation = (
                f"Selected top-scoring {phase.lower()} interventions linked to diagnosed risks: "
                f"{', '.join(top_risks) if top_risks else 'general support'}."
            )
        else:
            recommended_actions = [fallback_action]
            item_ids = []
            explanation = f"No catalog item was required for {phase}; fallback action keeps the 4-week path complete."

        return {
            "theme": phase,
            "objective": objective,
            "recommended_actions": recommended_actions,
            "expected_outcome": expected_outcome,
            "explanation": explanation,
            "item_ids": item_ids,
        }
