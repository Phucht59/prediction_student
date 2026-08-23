"""Top-K worklist by Hybrid p, then persistence action + route."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .contracts import (
    K_FRAC_PRIMARY,
    UNCERTAINTY_AUTO_MAX,
    PersistLabel,
    RecommendationDecision,
    RouteStatus,
    Stage,
)
from .feasibility import invalid_action, rule_label


def worklist_mask(frame: pd.DataFrame, *, k_frac: float = K_FRAC_PRIMARY) -> pd.Series:
    if not {"code_module", "code_presentation", "stage", "risk_probability"}.issubset(frame.columns):
        raise ValueError("worklist requires module, presentation, stage, risk_probability")
    selected = pd.Series(False, index=frame.index)
    ranks = pd.Series(np.nan, index=frame.index, dtype="float64")
    sizes = pd.Series(0, index=frame.index, dtype="int64")
    grouped = frame.groupby(["code_module", "code_presentation", "stage"], sort=False)
    for _, group in grouped:
        n = len(group)
        k = max(1, int(round(n * k_frac)))
        order = group["risk_probability"].astype(float).rank(method="first", ascending=False)
        in_top = order <= k
        selected.loc[group.index] = in_top.to_numpy()
        ranks.loc[group.index] = order.to_numpy()
        sizes.loc[group.index] = n
    frame = frame.copy()
    return selected, ranks.astype("Int64"), sizes


def attach_worklist(frame: pd.DataFrame, *, k_frac: float = K_FRAC_PRIMARY) -> pd.DataFrame:
    out = frame.copy()
    mask, ranks, sizes = worklist_mask(out, k_frac=k_frac)
    out["in_worklist"] = mask.to_numpy()
    out["rank_in_cohort"] = ranks.to_numpy()
    out["cohort_size"] = sizes.to_numpy()
    return out


def route_for_row(row: pd.Series, action: str) -> RouteStatus:
    if not bool(row.get("in_worklist", False)):
        return RouteStatus.OUT_OF_BUDGET
    if action == PersistLabel.COUNSEL.value:
        return RouteStatus.COUNSEL
    uncertainty = float(row.get("uncertainty") or 1.0)
    if uncertainty <= UNCERTAINTY_AUTO_MAX:
        return RouteStatus.ACTION
    return RouteStatus.QUEUE


def decision_from_row(
    row: pd.Series,
    *,
    action: str,
    score: float,
    pathway: tuple = (),
    stage: Stage | None = None,
) -> RecommendationDecision:
    if invalid_action(PersistLabel(action), row.to_dict(), stage):
        action = PersistLabel.COUNSEL.value
        score = 0.0
    route = route_for_row(row, action)
    if route is RouteStatus.OUT_OF_BUDGET:
        action = PersistLabel.COUNSEL.value
        reasons = ("OUT_OF_BUDGET_TOP_K",)
    elif route is RouteStatus.COUNSEL:
        reasons = ("NO_PERSISTING_LMS_LEVER",)
    elif route is RouteStatus.ACTION:
        reasons = ("WORKLIST_AND_CLEAR_PERSISTENCE",)
    else:
        reasons = ("WORKLIST_NEEDS_REVIEW",)
    stage_value = stage or Stage(str(row["stage"]))
    return RecommendationDecision(
        student_key=str(row.get("student_key") or row.get("id_student")),
        course_key=str(row.get("course_key") or f"{row.get('code_module')}::{row.get('code_presentation')}"),
        stage=stage_value,
        route=route,
        action=PersistLabel(action),
        score=float(np.clip(score, 0.0, 1.0)),
        reason_codes=reasons,
        pathway=pathway,
        in_worklist=bool(row.get("in_worklist", False)),
        rank_in_cohort=int(row["rank_in_cohort"]) if pd.notna(row.get("rank_in_cohort")) else None,
        cohort_size=int(row["cohort_size"]) if pd.notna(row.get("cohort_size")) else None,
    )


def rule_actions(frame: pd.DataFrame) -> np.ndarray:
    return np.array([rule_label(row.to_dict()).value for _, row in frame.iterrows()], dtype=object)


__all__ = [
    "attach_worklist",
    "decision_from_row",
    "route_for_row",
    "rule_actions",
    "worklist_mask",
]
