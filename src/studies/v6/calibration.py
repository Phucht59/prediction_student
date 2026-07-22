from __future__ import annotations

import json

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss

from src.studies.v5.common.metrics import expected_calibration_error

from .contract import ARTIFACT_ROOT, REPORT_ROOT, atomic_json, atomic_text, sha256_file


CALIBRATION_PATH = ARTIFACT_ROOT / "prediction/calibration.json"


def _logit(probability: np.ndarray) -> np.ndarray:
    values = np.clip(np.asarray(probability, dtype=float), 1e-6, 1 - 1e-6)
    return np.log(values / (1.0 - values))


def apply_temperature(probability: np.ndarray, temperature: float) -> np.ndarray:
    scaled = _logit(probability) / float(temperature)
    return 1.0 / (1.0 + np.exp(-scaled))


def _reliability(target: np.ndarray, probability: np.ndarray) -> list[dict[str, float | int]]:
    rows: list[dict[str, float | int]] = []
    for lower, upper in zip(np.linspace(0, 1, 11)[:-1], np.linspace(0, 1, 11)[1:]):
        selected = (probability >= lower) & (
            probability < upper if upper < 1 else probability <= upper
        )
        rows.append(
            {
                "lower": float(lower),
                "upper": float(upper),
                "count": int(selected.sum()),
                "mean_probability": float(probability[selected].mean()) if selected.any() else 0.0,
                "event_rate": float(target[selected].mean()) if selected.any() else 0.0,
            }
        )
    return rows


def _calibration_metrics(target: np.ndarray, probability: np.ndarray) -> dict[str, float]:
    return {
        "nll": float(log_loss(target, np.column_stack([1 - probability, probability]))),
        "brier": float(brier_score_loss(target, probability)),
        "ece": float(expected_calibration_error(target, probability)),
    }


def calibrate_final_predictions() -> dict:
    if CALIBRATION_PATH.is_file():
        cached = json.loads(CALIBRATION_PATH.read_text(encoding="utf-8"))
        if cached.get("status") == "COMPLETE" and (
            ARTIFACT_ROOT / "prediction/calibrated_oof_predictions.parquet"
        ).is_file():
            return cached
    inner = pd.read_parquet(ARTIFACT_ROOT / "prediction/multitask/inner_oof.parquet")
    inner = inner[inner.candidate.eq("W0")].copy()
    target = inner.target.to_numpy(dtype=int)
    probability = inner.probability.to_numpy(dtype=float)
    logits = _logit(probability)

    def objective(temperature: float) -> float:
        calibrated = apply_temperature(probability, temperature)
        return float(log_loss(target, np.column_stack([1 - calibrated, calibrated])))

    optimized = minimize_scalar(objective, bounds=(0.05, 10.0), method="bounded")
    if not optimized.success:
        raise RuntimeError(f"Temperature optimization failed: {optimized.message}")
    temperature = float(optimized.x)
    inner_calibrated = apply_temperature(probability, temperature)
    diagnostic = LogisticRegression(C=1e6, solver="lbfgs").fit(logits.reshape(-1, 1), target)

    seeds = pd.read_parquet(ARTIFACT_ROOT / "prediction/final/seed_predictions.parquet")
    seeds["calibrated_probability"] = apply_temperature(
        seeds.probability.to_numpy(dtype=float), temperature
    )
    keys = [
        "record_id",
        "id_student",
        "code_module",
        "code_presentation",
        "cutoff_day",
        "outer_fold",
        "target",
        "threshold",
    ]
    calibrated = seeds.groupby(keys, as_index=False).agg(
        probability_at_risk=("calibrated_probability", "mean"),
        seed_standard_deviation=("calibrated_probability", "std"),
        seed_minimum=("calibrated_probability", "min"),
        seed_maximum=("calibrated_probability", "max"),
    )
    calibrated["calibrated_threshold"] = apply_temperature(
        calibrated.threshold.to_numpy(dtype=float), temperature
    )
    p = np.clip(calibrated.probability_at_risk.to_numpy(dtype=float), 1e-12, 1 - 1e-12)
    calibrated["predictive_entropy"] = -(p * np.log(p) + (1 - p) * np.log(1 - p))
    calibrated["seed_disagreement"] = calibrated.seed_maximum - calibrated.seed_minimum
    calibrated.to_parquet(
        ARTIFACT_ROOT / "prediction/calibrated_oof_predictions.parquet", index=False
    )
    final_target = calibrated.target.to_numpy(dtype=int)
    final_probability = calibrated.probability_at_risk.to_numpy(dtype=float)
    result = {
        "schema_version": "v6_temperature_calibration_v1",
        "status": "COMPLETE",
        "method": "temperature_scaling",
        "temperature": temperature,
        "fit_scope": "candidate_C_W0_outer_training_fold_0_inner_oof_only",
        "fit_records": int(len(inner)),
        "fit_record_ids_sha256": sha256_file(
            ARTIFACT_ROOT / "prediction/multitask/inner_oof.parquet"
        ),
        "inner_before": _calibration_metrics(target, probability),
        "inner_after": _calibration_metrics(target, inner_calibrated),
        "calibration_slope_diagnostic": float(diagnostic.coef_[0, 0]),
        "calibration_intercept_diagnostic": float(diagnostic.intercept_[0]),
        "reliability_bins_inner_after": _reliability(target, inner_calibrated),
        "outer_reporting_only_after_freeze": _calibration_metrics(
            final_target, final_probability
        ),
        "outer_test_used_to_fit": False,
        "future_accessed": False,
    }
    atomic_json(CALIBRATION_PATH, result)
    atomic_text(
        REPORT_ROOT / "RANKING_AND_CALIBRATION_REPORT.md",
        f"""# V6 ranking and calibration report

Risk ranking did not pass the registered compatibility guardrails; Candidate C
remained frozen. Temperature scaling was fitted only on valid Candidate C/W0
inner-OOF predictions from outer-training fold 0.

- Temperature: {temperature:.6f}
- Inner NLL: {result['inner_before']['nll']:.6f} -> {result['inner_after']['nll']:.6f}
- Inner Brier: {result['inner_before']['brier']:.6f} -> {result['inner_after']['brier']:.6f}
- Inner ECE: {result['inner_before']['ece']:.6f} -> {result['inner_after']['ece']:.6f}
- Diagnostic slope/intercept: {result['calibration_slope_diagnostic']:.6f} / {result['calibration_intercept_diagnostic']:.6f}

Outer-fold values are reporting-only and never enter the calibrator fit.
""",
    )
    return result


__all__ = ["apply_temperature", "calibrate_final_predictions"]
