"""Risk stratification owned by the frozen Hybrid prediction authority."""

from __future__ import annotations

from .contracts import RecommendationFeatures, RiskBand, RiskThresholds


def stratify_risk(
    features: RecommendationFeatures,
    thresholds: RiskThresholds,
) -> RiskBand:
    """Return LOW, BORDERLINE, or HIGH without relearning risk.

    High model uncertainty or seed disagreement prevents automatic HIGH routing
    even when the mean probability is large. Such cases remain BORDERLINE and
    are monitored or reviewed rather than automatically recommended.
    """

    if features.risk_probability < thresholds.low:
        return RiskBand.LOW

    uncertain = (
        features.hybrid_uncertainty > thresholds.maximum_automatic_uncertainty
        or features.seed_disagreement > thresholds.maximum_seed_disagreement
    )
    if uncertain:
        return RiskBand.BORDERLINE

    if features.risk_probability >= thresholds.high:
        return RiskBand.HIGH

    return RiskBand.BORDERLINE


__all__ = ["stratify_risk"]
