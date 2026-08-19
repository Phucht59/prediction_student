"""C0-aligned risk routing. Uses frozen prediction threshold t, never H1 0.2/0.8."""

from __future__ import annotations

from .contracts import RecommendationFeatures, RiskRoute, RiskThresholds


def stratify_risk(features: RecommendationFeatures, thresholds: RiskThresholds) -> RiskRoute:
    """Route using p vs C0 t plus recommendation-only uncertainty/margin caps.

    None-safe: missing optional signals never raise.
    """

    if features.risk_probability < features.prediction_threshold:
        return RiskRoute.NO_AUTOMATIC
    uncertain = features.uncertainty > thresholds.maximum_automatic_uncertainty
    thin_margin = features.risk_margin < thresholds.minimum_risk_margin
    if uncertain or thin_margin:
        return RiskRoute.HUMAN_REVIEW
    return RiskRoute.PROCESS
