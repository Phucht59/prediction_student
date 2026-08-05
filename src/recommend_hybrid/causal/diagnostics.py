"""Diagnostics and identifiability gates for observational target trials."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

EPSILON = 1.0e-9


def _as_binary(values: np.ndarray, name: str) -> np.ndarray:
    result = np.asarray(values, dtype=np.int8).reshape(-1)
    if not np.isin(result, [0, 1]).all():
        raise ValueError(f"{name} must be binary")
    return result


def _weighted_mean_variance(
    values: np.ndarray,
    weights: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    denominator = np.sum(weights)
    if denominator <= 0.0:
        raise ValueError("weights must have positive mass")
    mean = np.sum(values * weights[:, None], axis=0) / denominator
    variance = np.sum(weights[:, None] * (values - mean) ** 2, axis=0) / denominator
    return mean, variance


def standardized_mean_difference(
    features: np.ndarray,
    treatment: np.ndarray,
    weights: np.ndarray | None = None,
) -> np.ndarray:
    """Return absolute SMD for every baseline feature."""

    x = np.asarray(features, dtype=np.float64)
    if x.ndim == 1:
        x = x[:, None]
    t = _as_binary(treatment, "treatment")
    if len(x) != len(t):
        raise ValueError("features and treatment must align")
    w = np.ones(len(t), dtype=np.float64) if weights is None else np.asarray(
        weights, dtype=np.float64
    ).reshape(-1)
    if len(w) != len(t) or not np.isfinite(w).all() or (w < 0.0).any():
        raise ValueError("weights must be finite, non-negative, and aligned")

    treated = t == 1
    control = ~treated
    if not treated.any() or not control.any():
        return np.full(x.shape[1], np.inf, dtype=np.float64)
    mean_t, var_t = _weighted_mean_variance(x[treated], w[treated])
    mean_c, var_c = _weighted_mean_variance(x[control], w[control])
    pooled = np.sqrt(np.maximum(0.5 * (var_t + var_c), EPSILON))
    return np.abs(mean_t - mean_c) / pooled


def effective_sample_size(weights: np.ndarray) -> float:
    values = np.asarray(weights, dtype=np.float64).reshape(-1)
    if not len(values) or not np.isfinite(values).all() or (values < 0.0).any():
        raise ValueError("weights must be finite and non-negative")
    denominator = float(np.sum(values**2))
    return float(np.sum(values) ** 2 / denominator) if denominator > 0.0 else 0.0


def stabilized_iptw(
    treatment: np.ndarray,
    propensity: np.ndarray,
    *,
    overlap_bounds: tuple[float, float] = (0.10, 0.90),
    clip_quantiles: tuple[float, float] = (0.01, 0.99),
) -> tuple[np.ndarray, np.ndarray]:
    """Build stabilized IPTW after overlap trimming and weight clipping."""

    t = _as_binary(treatment, "treatment")
    p = np.asarray(propensity, dtype=np.float64).reshape(-1)
    if len(t) != len(p) or not np.isfinite(p).all():
        raise ValueError("propensity must be finite and aligned")
    low, high = map(float, overlap_bounds)
    if not 0.0 < low < high < 1.0:
        raise ValueError("overlap bounds must lie strictly inside (0, 1)")
    keep = (p >= low) & (p <= high)
    if not keep.any():
        return np.zeros_like(p), keep

    prevalence = float(np.mean(t[keep]))
    clipped_p = np.clip(p, EPSILON, 1.0 - EPSILON)
    weights = np.zeros_like(clipped_p)
    weights[keep & (t == 1)] = prevalence / clipped_p[keep & (t == 1)]
    weights[keep & (t == 0)] = (1.0 - prevalence) / (
        1.0 - clipped_p[keep & (t == 0)]
    )
    active = weights[keep]
    q_low, q_high = np.quantile(active, clip_quantiles)
    weights[keep] = np.clip(active, q_low, q_high)
    return weights, keep


@dataclass(frozen=True)
class IdentifiabilityThresholds:
    minimum_treated: int = 30
    minimum_control: int = 30
    overlap_bounds: tuple[float, float] = (0.10, 0.90)
    maximum_good_smd: float = 0.10
    maximum_allowed_smd: float = 0.20
    minimum_good_feature_fraction: float = 0.90
    minimum_ess_fraction: float = 0.25
    maximum_trim_fraction: float = 0.30


@dataclass(frozen=True)
class IdentifiabilityReport:
    identifiable: bool
    reasons: tuple[str, ...]
    sample_count: int
    retained_count: int
    treated_count: int
    control_count: int
    trim_fraction: float
    effective_sample_size: float
    ess_fraction: float
    good_smd_fraction: float
    maximum_smd: float

    def to_dict(self) -> dict[str, object]:
        return {
            "identifiable": self.identifiable,
            "reasons": list(self.reasons),
            "sample_count": self.sample_count,
            "retained_count": self.retained_count,
            "treated_count": self.treated_count,
            "control_count": self.control_count,
            "trim_fraction": self.trim_fraction,
            "effective_sample_size": self.effective_sample_size,
            "ess_fraction": self.ess_fraction,
            "good_smd_fraction": self.good_smd_fraction,
            "maximum_smd": self.maximum_smd,
        }


def assess_identifiability(
    *,
    features: np.ndarray,
    treatment: np.ndarray,
    propensity: np.ndarray,
    thresholds: IdentifiabilityThresholds = IdentifiabilityThresholds(),
) -> tuple[IdentifiabilityReport, np.ndarray, np.ndarray]:
    """Run the preregistered overlap, balance, count, and ESS gates."""

    x = np.asarray(features, dtype=np.float64)
    if x.ndim == 1:
        x = x[:, None]
    t = _as_binary(treatment, "treatment")
    weights, keep = stabilized_iptw(
        t,
        propensity,
        overlap_bounds=thresholds.overlap_bounds,
    )
    retained = int(keep.sum())
    treated_count = int(np.sum(t[keep] == 1))
    control_count = int(np.sum(t[keep] == 0))
    trim_fraction = 1.0 - retained / len(t) if len(t) else 1.0
    ess = effective_sample_size(weights[keep]) if retained else 0.0
    ess_fraction = ess / retained if retained else 0.0
    smd = standardized_mean_difference(x[keep], t[keep], weights[keep]) if retained else np.array([np.inf])
    finite_smd = smd[np.isfinite(smd)]
    maximum_smd = float(np.max(finite_smd)) if len(finite_smd) else float("inf")
    good_fraction = float(np.mean(smd < thresholds.maximum_good_smd)) if len(smd) else 0.0

    reasons: list[str] = []
    if treated_count < thresholds.minimum_treated:
        reasons.append("INSUFFICIENT_TREATED")
    if control_count < thresholds.minimum_control:
        reasons.append("INSUFFICIENT_CONTROL")
    if trim_fraction > thresholds.maximum_trim_fraction:
        reasons.append("INSUFFICIENT_OVERLAP")
    if ess_fraction < thresholds.minimum_ess_fraction:
        reasons.append("LOW_EFFECTIVE_SAMPLE_SIZE")
    if good_fraction < thresholds.minimum_good_feature_fraction:
        reasons.append("COVARIATE_BALANCE_NOT_ACHIEVED")
    if maximum_smd > thresholds.maximum_allowed_smd:
        reasons.append("EXTREME_RESIDUAL_IMBALANCE")

    report = IdentifiabilityReport(
        identifiable=not reasons,
        reasons=tuple(reasons),
        sample_count=len(t),
        retained_count=retained,
        treated_count=treated_count,
        control_count=control_count,
        trim_fraction=float(trim_fraction),
        effective_sample_size=float(ess),
        ess_fraction=float(ess_fraction),
        good_smd_fraction=float(good_fraction),
        maximum_smd=float(maximum_smd),
    )
    return report, weights, keep


__all__ = [
    "IdentifiabilityReport",
    "IdentifiabilityThresholds",
    "assess_identifiability",
    "effective_sample_size",
    "stabilized_iptw",
    "standardized_mean_difference",
]
