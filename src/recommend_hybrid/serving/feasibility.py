"""Hard feasibility for the three serving actions. Fail-closed."""

from __future__ import annotations

import pandas as pd

from .contracts import (
    ACTIVE_DAY_ENGAGE,
    STREAK_ENGAGE,
    PersistLabel,
    Stage,
)


def _finite_int(value, default: int = 0) -> int:
    if value is None:
        return default
    try:
        if pd.isna(value):
            return default
    except (TypeError, ValueError):
        pass
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def _finite_float(value, default: float | None = None) -> float | None:
    if value is None:
        return default
    try:
        if pd.isna(value):
            return default
    except (TypeError, ValueError):
        pass
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def assess_eligible(row: dict) -> bool:
    missing = _finite_int(row.get("missing_assessment_count"), 0)
    due_soon = _finite_int(row.get("due_soon_count"), 0)
    remaining = _finite_int(row.get("remaining_count"), 0)
    return (missing >= 1 or due_soon >= 2) and remaining >= 0


def engage_eligible(row: dict) -> bool:
    if not bool(row.get("vle_access_available", False)):
        return False
    streak = _finite_int(row.get("inactivity_streak"), 0)
    active = _finite_float(row.get("active_day_rate"), 1.0)
    streak_ok = streak >= STREAK_ENGAGE
    active_ok = active is not None and active < ACTIVE_DAY_ENGAGE
    return bool(streak_ok or active_ok)


def feasible_labels(row: dict) -> tuple[PersistLabel, ...]:
    labels: list[PersistLabel] = []
    if assess_eligible(row):
        labels.append(PersistLabel.ASSESS)
    if engage_eligible(row):
        labels.append(PersistLabel.ENGAGE)
    labels.append(PersistLabel.COUNSEL)
    return tuple(labels)


def rule_label(row: dict) -> PersistLabel:
    """Deterministic tail baseline. Same priority as serving decoder."""
    if assess_eligible(row):
        return PersistLabel.ASSESS
    if engage_eligible(row):
        return PersistLabel.ENGAGE
    return PersistLabel.COUNSEL


def invalid_action(label: PersistLabel, row: dict, stage: Stage | str | None = None) -> bool:
    if label is PersistLabel.COUNSEL:
        return False
    if label is PersistLabel.ASSESS:
        remaining = int(row.get("remaining_count") or 0)
        missing = int(row.get("missing_assessment_count") or 0)
        due_soon = int(row.get("due_soon_count") or 0)
        if missing < 1 and due_soon < 2:
            return True
        if remaining < 0:
            return True
        return False
    if label is PersistLabel.ENGAGE:
        return not engage_eligible(row)
    return True


__all__ = ["assess_eligible", "engage_eligible", "feasible_labels", "invalid_action", "rule_label"]
