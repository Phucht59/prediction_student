"""Build a compact semantic state from adapter output and feature frames."""

from __future__ import annotations

import pandas as pd

from ..contracts.state import make_case_id, operational_risk_band


JOIN_KEYS = ["dataset", "student_id", "record_id", "stage", "outer_fold"]


class StudentStateBuilder:
    """Join prediction authority to already-audited, record-aligned features."""

    def __init__(self, *, low_risk_max: float = 0.33, medium_risk_max: float = 0.66) -> None:
        self.low_risk_max = low_risk_max
        self.medium_risk_max = medium_risk_max

    def build(self, predictions: pd.DataFrame, features: pd.DataFrame) -> pd.DataFrame:
        prediction_required = set(JOIN_KEYS + ["risk_probability", "prediction_source_version"])
        missing_predictions = prediction_required.difference(predictions)
        missing_features = set(JOIN_KEYS).difference(features)
        if missing_predictions or missing_features:
            raise ValueError({"prediction_missing": sorted(missing_predictions), "feature_missing": sorted(missing_features)})
        if predictions.duplicated(["record_id", "stage"]).any():
            raise ValueError("prediction rows are not unique by record-stage")
        if features.duplicated(["record_id", "stage"]).any():
            raise ValueError("feature rows are not unique by record-stage")
        forbidden = {"target", "final_result", "score", "date_unregistration"}.intersection(features.columns)
        if forbidden:
            raise ValueError(f"forbidden source fields reached StudentStateBuilder: {sorted(forbidden)}")
        joined = predictions.merge(
            features, on=JOIN_KEYS, how="inner", validate="one_to_one", suffixes=("", "_feature")
        )
        if len(joined) != len(predictions):
            raise ValueError("prediction-to-feature join is incomplete")
        joined["case_id"] = [make_case_id(d, r, s) for d, r, s in zip(joined.dataset, joined.record_id, joined.stage, strict=True)]
        joined["risk_band"] = [
            operational_risk_band(float(value), low_risk_max=self.low_risk_max, medium_risk_max=self.medium_risk_max)
            for value in joined["risk_probability"]
        ]
        joined["recommendation_eligible"] = True
        joined["risk_band_source"] = "operational_config_not_prediction_threshold"
        columns = [
            "case_id", "dataset", "student_id", "record_id", "stage", "outer_fold",
            "risk_probability", "risk_band", "prediction_source_version", "prediction_seed_count", "risk_band_source",
            "recommendation_eligible",
        ]
        if "uncertainty" in joined and joined["uncertainty"].notna().any():
            columns.insert(8, "uncertainty")
        feature_columns = [
            "module", "presentation", "enrollment_identity",
            "inactive_streak", "active_days_ratio", "recent_activity", "activity_trend",
            "assessment_completion", "missing_assessments", "course_progress", "quiz_activity",
            "vle_available", "source_feature_version",
        ]
        return joined[columns + [c for c in feature_columns if c in joined]].sort_values(
            ["stage", "record_id"], kind="mergesort"
        ).reset_index(drop=True)
