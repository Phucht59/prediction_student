"""Serve persistence recommendation on frozen Hybrid probabilities."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.prediction.contracts import PredictionResult

from .contracts import (
    K_FRAC_PRIMARY,
    PROTOCOL_VERSION,
    RecommendationDecision,
    RouteStatus,
    Stage,
    map_prediction_state,
)
from .model import PersistenceClassifier
from .policy import attach_worklist, decision_from_row


class PersistencePipeline:
    def __init__(
        self,
        classifier: PersistenceClassifier,
        cohort: pd.DataFrame,
        *,
        k_frac: float = K_FRAC_PRIMARY,
    ) -> None:
        if "query_id" not in cohort.columns:
            raise ValueError("cohort requires query_id")
        self.classifier = classifier
        self.k_frac = k_frac
        self.cohort = attach_worklist(cohort, k_frac=k_frac)

    @classmethod
    def from_artifacts(cls, model_path: Path, cohort_path: Path, *, k_frac: float = K_FRAC_PRIMARY):
        return cls(PersistenceClassifier.load(model_path), pd.read_parquet(cohort_path), k_frac=k_frac)

    def _row_for_query(self, query_id: str, result: PredictionResult | None = None) -> pd.Series:
        matched = self.cohort.loc[self.cohort["query_id"].astype(str) == str(query_id)]
        if matched.empty:
            raise KeyError(query_id)
        row = matched.iloc[0].copy()
        if result is not None:
            if result.model_id != "hybrid":
                raise ValueError("only model_id='hybrid' is accepted")
            row["risk_probability"] = float(result.risk_probability)
            row["prediction_threshold"] = float(result.threshold)
            row["predicted_risk"] = int(result.predicted_risk)
            if result.uncertainty is not None:
                row["uncertainty"] = float(result.uncertainty)
        return row

    def recommend_query(self, query_id: str, result: PredictionResult | None = None) -> RecommendationDecision:
        row = self._row_for_query(query_id, result)
        frame = pd.DataFrame([row])
        actions, scores = self.classifier.constrained_predict(frame)
        stage = Stage(str(row["stage"]))
        return decision_from_row(row, action=str(actions[0]), score=float(scores[0]), stage=stage)

    def recommend_result(self, result: PredictionResult, query_id: str) -> RecommendationDecision:
        if result.stage_or_endpoint in {"100pct", "100", "FINAL-100"}:
            raise ValueError("100pct is not an intervention stage")
        _ = map_prediction_state(result.stage_or_endpoint)
        return self.recommend_query(query_id, result)


__all__ = ["PersistencePipeline", "PROTOCOL_VERSION", "RouteStatus"]
