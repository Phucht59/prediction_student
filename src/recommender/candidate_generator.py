import pandas as pd
from typing import Any


class CandidateGenerator:
    def __init__(self, catalog_df: pd.DataFrame):
        self.catalog_df = catalog_df

    @staticmethod
    def _risk_threshold(predicted_class: int, class_probabilities: list[float] | None = None) -> float:
        """Use a prediction-aware threshold for activating risks."""
        if class_probabilities is None or len(class_probabilities) < 3:
            return {0: 0.25, 1: 0.35, 2: 0.45}.get(int(predicted_class), 0.35)

        p_low, p_medium, p_high = [float(x) for x in class_probabilities[:3]]
        if int(predicted_class) == 0:
            # For likely Low students, be sensitive: missing a real risk is worse
            # than recommending one extra support action.
            return max(0.20, 0.35 - 0.15 * p_low)
        if int(predicted_class) == 1:
            uncertainty = 1.0 - max(p_low, p_medium, p_high)
            return max(0.28, 0.38 - 0.10 * uncertainty)
        # Stable/high students should not receive heavy remedial interventions
        # unless a risk probability is genuinely high.
        return 0.50 if p_high >= 0.60 else 0.42

    def generate_candidates(
        self,
        diagnosed_risks: dict[str, float],
        predicted_class: int,
        class_probabilities: list[float] | None = None,
        min_candidates: int = 5,
    ) -> pd.DataFrame:
        """
        Filter intervention catalog items before scoring.

        The filter is tied to the prediction result: Low/uncertain students use
        a lower risk threshold to prioritize early support, while High students
        receive remedial items only when risk probabilities are strong.
        """
        threshold = self._risk_threshold(predicted_class, class_probabilities)
        active_risks = [risk for risk, prob in diagnosed_risks.items() if float(prob) >= threshold]
        strong_risks = [risk for risk, prob in diagnosed_risks.items() if float(prob) >= 0.60]

        # If the model predicts Low but the risk head is conservative, keep the
        # strongest diagnosed risk so that the learning path remains actionable.
        if not active_risks and int(predicted_class) == 0 and diagnosed_risks:
            active_risks = [max(diagnosed_risks.items(), key=lambda item: item[1])[0]]

        candidates = []
        seen_item_ids: set[str] = set()

        for _, row in self.catalog_df.iterrows():
            item_id = str(row.get("item_id", ""))
            if item_id in seen_item_ids:
                continue

            target_risks_str = str(row.get("target_risks", ""))
            if pd.isna(row.get("target_risks")):
                target_risks_str = ""
            target_risks = [r.strip() for r in target_risks_str.split(",") if r.strip()]

            keep = False
            if not target_risks:
                # Enrichment/general items are mainly for stable students.
                keep = int(predicted_class) == 2 and not strong_risks
            elif any(r in active_risks for r in target_risks):
                keep = True
            elif int(predicted_class) == 0 and any(
                r in {"R1_LOW_PRIOR_PERFORMANCE", "R3_ATTENDANCE_RISK", "R4_LOW_ENGAGEMENT", "R6_HIGH_FAILURE_PROBABILITY"}
                for r in target_risks
            ):
                # Low prediction keeps core intervention families available even
                # when a single risk score is just below threshold.
                keep = True
            elif int(predicted_class) == 1 and any(r in strong_risks for r in target_risks):
                keep = True

            if keep:
                candidates.append(row)
                seen_item_ids.add(item_id)

        if len(candidates) < min_candidates:
            # Fill with broadly useful low-difficulty items before falling back to
            # the whole catalog. This avoids sparse recommendations without
            # flooding the scorer with advanced/remedial items.
            filler = self.catalog_df.sort_values(
                by=["difficulty_level", "expected_effect"], ascending=[True, False]
            )
            for _, row in filler.iterrows():
                item_id = str(row.get("item_id", ""))
                if item_id not in seen_item_ids:
                    candidates.append(row)
                    seen_item_ids.add(item_id)
                if len(candidates) >= min_candidates:
                    break

        if not candidates:
            return self.catalog_df.copy()
        return pd.DataFrame(candidates)
