"""Constrained same-stage behavioural recourse through the frozen OULAD path."""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Callable, Mapping

import numpy as np

from src.pipelines import oulad

from .taxonomy import LEARNED_ACTIONS


class SimulationStrength(str, Enum):
    CONSERVATIVE = "conservative"
    MODERATE = "moderate"
    STRONG = "strong"


STRENGTH_INDEX = {
    SimulationStrength.CONSERVATIVE: 0,
    SimulationStrength.MODERATE: 1,
    SimulationStrength.STRONG: 2,
}


@dataclass(frozen=True)
class SimulatedStageInputs:
    full_sequence: np.ndarray
    raw_aggregate: np.ndarray
    base_sequence: np.ndarray
    constraint_violations: tuple[str, ...]


ACTION_PARAMETERS: Mapping[str, Mapping[str, tuple[float, float, float]]] = {
    "VLE_ENGAGEMENT": {
        "click_multiplier": (1.10, 1.25, 1.50),
        "minimum_clicks": (3.0, 7.0, 12.0),
        "active_day_increment": (1.0, 2.0, 3.0),
        "site_increment": (1.0, 2.0, 3.0),
    },
    "STUDY_REGULARITY": {
        "minimum_clicks": (3.0, 7.0, 12.0),
        "minimum_active_days": (1.0, 2.0, 3.0),
        "minimum_sites": (1.0, 2.0, 3.0),
    },
    "QUIZ_OR_RETRIEVAL_PRACTICE": {
        "quiz_click_increment": (3.0, 7.0, 12.0),
    },
    "CONTENT_REVIEW": {
        "content_click_increment": (5.0, 12.0, 20.0),
    },
    "ASSESSMENT_COMPLETION": {
        "submission_increment": (1.0, 1.0, 1.0),
        "assessment_click_increment": (3.0, 7.0, 12.0),
    },
}


def _parameter(action_id: str, name: str, strength: SimulationStrength) -> float:
    try:
        return float(ACTION_PARAMETERS[action_id][name][STRENGTH_INDEX[strength]])
    except KeyError as error:
        raise ValueError(f"unsupported simulation parameter {action_id}/{name}") from error


def _validate_inputs(
    full_sequence: np.ndarray,
    lengths: np.ndarray,
    context: np.ndarray,
    applicable: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    sequence = np.asarray(full_sequence, dtype=np.float32)
    length = np.asarray(lengths, dtype=np.int64).reshape(-1)
    stage_context = np.asarray(context, dtype=np.float32)
    app = np.asarray(applicable, dtype=bool).reshape(-1)
    if sequence.ndim != 3 or sequence.shape[2] != len(oulad.CHANNELS):
        raise ValueError("full_sequence must be [rows, weeks, 47]")
    if len(length) != len(sequence) or len(app) != len(sequence):
        raise ValueError("lengths/applicable must align with sequence rows")
    if stage_context.shape != (len(sequence), len(oulad.CONTEXT_COLUMNS)):
        raise ValueError("context must contain the four immutable stage-context features")
    if np.any(length <= 0) or np.any(length > sequence.shape[1]):
        raise ValueError("lengths must identify observed sequence positions")
    if not np.isfinite(sequence).all() or not np.isfinite(stage_context).all():
        raise ValueError("simulation inputs must be finite")
    return sequence, length, stage_context, app


def _recompute_inactivity(
    base: np.ndarray,
    lengths: np.ndarray,
    applicable: np.ndarray,
) -> None:
    index = {name: position for position, name in enumerate(oulad.BASE_CHANNELS)}
    for row, length in enumerate(lengths):
        if not applicable[row]:
            continue
        streak = 0
        days = 0
        for week in range(int(length)):
            active = base[row, week, index["total_clicks"]] > 0.0
            if active:
                streak = 0
                days = 0
            else:
                streak += 1
                days += 7
            base[row, week, index["weeks_without_activity"]] = float(streak)
            base[row, week, index["days_since_last_vle_activity"]] = float(days)


def _enforce_constraints(base: np.ndarray, lengths: np.ndarray) -> tuple[str, ...]:
    index = {name: position for position, name in enumerate(oulad.BASE_CHANNELS)}
    violations: list[str] = []
    base[:] = np.maximum(base, 0.0)
    base[:, :, index["active_days"]] = np.minimum(base[:, :, index["active_days"]], 7.0)
    component_names = (
        "content_clicks",
        "forum_clicks",
        "quiz_clicks",
        "assessment_related_clicks",
    )
    for row, length in enumerate(lengths):
        observed = slice(0, int(length))
        total = base[row, observed, index["total_clicks"]]
        for name in component_names:
            component = base[row, observed, index[name]]
            if np.any(component > total + 1.0e-6):
                violations.append(f"{row}:{name}_EXCEEDS_TOTAL_CLICKS")
                base[row, observed, index[name]] = np.minimum(component, total)
        if np.any(base[row, int(length) :, :] != 0.0):
            violations.append(f"{row}:FUTURE_PADDING_MODIFIED")
            base[row, int(length) :, :] = 0.0
    return tuple(sorted(set(violations)))


def simulate_action_inputs(
    *,
    full_sequence: np.ndarray,
    lengths: np.ndarray,
    stage_context: np.ndarray,
    action_id: str,
    strength: SimulationStrength | str,
    applicable: np.ndarray,
    recent_weeks: int = 2,
) -> SimulatedStageInputs:
    """Edit current observed behaviour and rebuild all derived Hybrid inputs.

    This is same-stage recourse sensitivity: it asks how the frozen model would
    respond if the current observed behaviour indicators were better.  It is
    not a post-recommendation trajectory and must not be interpreted causally.
    """

    if action_id not in LEARNED_ACTIONS:
        raise ValueError(f"action {action_id!r} is not a learned behavioural action")
    resolved_strength = SimulationStrength(strength)
    sequence, length, context, app = _validate_inputs(
        full_sequence,
        lengths,
        stage_context,
        applicable,
    )
    base = sequence[:, :, : len(oulad.BASE_CHANNELS)].copy()
    index = {name: position for position, name in enumerate(oulad.BASE_CHANNELS)}

    for row, observed_length in enumerate(length):
        if not app[row]:
            continue
        end = int(observed_length)
        start = max(0, end - int(recent_weeks))
        selected = slice(start, end)
        if action_id == "VLE_ENGAGEMENT":
            multiplier = _parameter(action_id, "click_multiplier", resolved_strength)
            minimum = _parameter(action_id, "minimum_clicks", resolved_strength)
            increment = _parameter(action_id, "active_day_increment", resolved_strength)
            sites = _parameter(action_id, "site_increment", resolved_strength)
            original_total = base[row, selected, index["total_clicks"]].copy()
            inactive = original_total <= 0.0
            total = original_total * multiplier
            total[inactive] = minimum
            base[row, selected, index["total_clicks"]] = total
            content = base[row, selected, index["content_clicks"]] * multiplier
            content[inactive] = minimum * 0.6
            base[row, selected, index["content_clicks"]] = content
            base[row, selected, index["active_days"]] = np.maximum(
                base[row, selected, index["active_days"]] + increment,
                np.where(inactive, 1.0, 0.0),
            )
            base[row, selected, index["unique_sites"]] += sites
            base[row, selected, index["unique_activity_types"]] += 1.0
        elif action_id == "STUDY_REGULARITY":
            minimum_clicks = _parameter(action_id, "minimum_clicks", resolved_strength)
            minimum_days = _parameter(action_id, "minimum_active_days", resolved_strength)
            minimum_sites = _parameter(action_id, "minimum_sites", resolved_strength)
            total = base[row, selected, index["total_clicks"]]
            inactive = total <= 0.0
            total[inactive] = minimum_clicks
            base[row, selected, index["total_clicks"]] = total
            active_days = base[row, selected, index["active_days"]]
            active_days[inactive] = np.maximum(active_days[inactive], minimum_days)
            base[row, selected, index["active_days"]] = active_days
            sites = base[row, selected, index["unique_sites"]]
            sites[inactive] = np.maximum(sites[inactive], minimum_sites)
            base[row, selected, index["unique_sites"]] = sites
            content = base[row, selected, index["content_clicks"]]
            content[inactive] = np.maximum(content[inactive], minimum_clicks * 0.6)
            base[row, selected, index["content_clicks"]] = content
        elif action_id == "QUIZ_OR_RETRIEVAL_PRACTICE":
            increment = _parameter(action_id, "quiz_click_increment", resolved_strength)
            base[row, selected, index["quiz_clicks"]] += increment
            base[row, selected, index["assessment_related_clicks"]] += increment
            base[row, selected, index["total_clicks"]] += increment
            base[row, selected, index["active_days"]] = np.maximum(
                base[row, selected, index["active_days"]],
                1.0,
            )
        elif action_id == "CONTENT_REVIEW":
            increment = _parameter(action_id, "content_click_increment", resolved_strength)
            base[row, selected, index["content_clicks"]] += increment
            base[row, selected, index["total_clicks"]] += increment
            base[row, selected, index["active_days"]] = np.maximum(
                base[row, selected, index["active_days"]],
                1.0,
            )
            base[row, selected, index["unique_sites"]] += 1.0
        elif action_id == "ASSESSMENT_COMPLETION":
            last_week = end - 1
            submission = _parameter(action_id, "submission_increment", resolved_strength)
            clicks = _parameter(action_id, "assessment_click_increment", resolved_strength)
            base[row, last_week, index["submitted_assessment_count"]] += submission
            base[row, last_week, index["assessment_related_clicks"]] += clicks
            base[row, last_week, index["total_clicks"]] += clicks
            base[row, last_week, index["active_days"]] = max(
                base[row, last_week, index["active_days"]],
                1.0,
            )

    _recompute_inactivity(base, length, app)
    violations = _enforce_constraints(base, length)
    mask = np.arange(sequence.shape[1])[None, :] < length[:, None]
    full = oulad._dynamic(base, mask)
    aggregate = oulad._aggregate(base, length)
    raw_aggregate = np.column_stack([aggregate, context]).astype(np.float32)
    if full.shape != sequence.shape or raw_aggregate.shape[1] != 165:
        raise RuntimeError("simulated OULAD feature contract changed")
    return SimulatedStageInputs(full, raw_aggregate, base, violations)


def predict_risk_sensitivity(
    *,
    baseline_risk: np.ndarray,
    simulated_inputs: Mapping[SimulationStrength, SimulatedStageInputs],
    predictor: Callable[[SimulatedStageInputs], np.ndarray],
) -> dict[str, object]:
    """Evaluate model response without interpreting it as a causal effect."""

    baseline = np.asarray(baseline_risk, dtype=np.float64).reshape(-1)
    rows: dict[str, object] = {}
    previous: np.ndarray | None = None
    monotonic = np.ones(len(baseline), dtype=bool)
    all_violations: list[str] = []
    for strength in SimulationStrength:
        inputs = simulated_inputs[strength]
        risk = np.asarray(predictor(inputs), dtype=np.float64).reshape(-1)
        if len(risk) != len(baseline) or not np.isfinite(risk).all():
            raise ValueError("predictor returned invalid simulated risk")
        if np.any((risk < 0.0) | (risk > 1.0)):
            raise ValueError("predictor must return probabilities")
        if previous is not None:
            monotonic &= risk <= previous + 1.0e-8
        previous = risk
        delta = baseline - risk
        rows[strength.value] = {
            "mean_risk_delta": float(delta.mean()),
            "median_risk_delta": float(np.median(delta)),
            "positive_reduction_fraction": float(np.mean(delta > 0.0)),
            "simulated_risk": risk,
        }
        all_violations.extend(inputs.constraint_violations)
    return {
        "strengths": rows,
        "monotonic_strength_fraction": float(monotonic.mean()),
        "constraint_violation_count": len(set(all_violations)),
        "constraint_violations": sorted(set(all_violations)),
        "claim_boundary": "SAME_STAGE_MODEL_RECOURSE_NOT_CAUSAL_EFFECT",
    }


__all__ = [
    "ACTION_PARAMETERS",
    "SimulatedStageInputs",
    "SimulationStrength",
    "predict_risk_sensitivity",
    "simulate_action_inputs",
]
