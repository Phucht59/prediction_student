"""Panel sampling from reconciled Student State only."""

from __future__ import annotations

import hashlib

import pandas as pd


PANEL_BANDS = ("Low", "Borderline", "High")
STAGES = ("20pct", "35pct", "50pct", "75pct")
FOLDS = (0, 1, 2)
BAND_MAP = {"low": "Low", "medium": "Borderline", "high": "High"}


def _order(seed: int, panel: str, student_id: str, case_id: str) -> int:
    value = f"{seed}|{panel}|{student_id}|{case_id}".encode("utf-8")
    return int(hashlib.sha256(value).hexdigest()[:16], 16)


def _allocate(available: dict[str, int], target: int) -> dict[str, int]:
    active = sorted(key for key, value in available.items() if value > 0)
    if target < len(active):
        raise ValueError("target is too small to cover all available strata")
    quotas = {key: 1 for key in active}
    remaining = target - len(active)
    if not remaining:
        return quotas
    capacity = {key: available[key] - 1 for key in active}
    total = sum(capacity.values())
    if total <= 0:
        return quotas
    raw = {key: remaining * capacity[key] / total for key in active}
    for key in active:
        quotas[key] += min(capacity[key], int(raw[key]))
    left = target - sum(quotas.values())
    order = sorted(active, key=lambda key: (raw[key] - int(raw[key]), capacity[key], key), reverse=True)
    while left:
        changed = False
        for key in order:
            if quotas[key] < available[key]:
                quotas[key] += 1
                left -= 1
                changed = True
                if not left: break
        if not changed:
            break
    if sum(quotas.values()) != target:
        raise ValueError("stratum allocation could not satisfy target")
    return quotas


def _prepare(state: pd.DataFrame, seed: int, panel: str, excluded_students: set[str]) -> pd.DataFrame:
    required = {"case_id", "student_id", "enrollment_identity", "stage", "outer_fold", "risk_band", "risk_probability"}
    missing = required.difference(state.columns)
    if missing: raise ValueError(f"state missing sampling columns: {sorted(missing)}")
    work = state[state.stage.isin(STAGES) & ~state.student_id.astype(str).isin(excluded_students)].copy()
    work["student_id"] = work["student_id"].astype(str)
    work["sampling_risk_band"] = work["risk_band"].astype(str).str.casefold().map(BAND_MAP)
    if work.sampling_risk_band.isna().any(): raise ValueError("state contains an unknown risk band")
    work["stratum"] = work.stage.astype(str) + "|" + work.outer_fold.astype(str) + "|" + work.sampling_risk_band
    work["_order"] = [_order(seed, panel, s, c) for s, c in zip(work.student_id, work.case_id, strict=True)]
    return work.sort_values(["stratum", "_order"], kind="mergesort")


def sample_panel(
    state: pd.DataFrame,
    *,
    panel: str,
    target_size: int,
    seed: int = 2026,
    excluded_students: set[str] | None = None,
) -> pd.DataFrame:
    """Select one case per learner, with deterministic 36-stratum allocation."""
    if panel not in {"A", "B"}: raise ValueError("panel must be A or B")
    excluded_students = set(excluded_students or set())
    work = _prepare(state, seed, panel, excluded_students)
    candidates = work.drop_duplicates(["stratum", "student_id"], keep="first")
    available = candidates.groupby("stratum").student_id.nunique().astype(int).to_dict()
    quotas = _allocate(available, target_size)
    chosen_rows = []
    chosen_students: set[str] = set(excluded_students)
    for stratum in sorted(quotas):
        quota = quotas[stratum]
        for row in candidates[candidates.stratum == stratum].itertuples(index=False):
            if row.student_id in chosen_students: continue
            chosen_rows.append(row._asdict())
            chosen_students.add(row.student_id)
            quota -= 1
            if quota == 0: break
    if len(chosen_rows) < target_size:
        for row in work.itertuples(index=False):
            if row.student_id in chosen_students: continue
            chosen_rows.append(row._asdict())
            chosen_students.add(row.student_id)
            if len(chosen_rows) == target_size: break
    if len(chosen_rows) != target_size:
        raise ValueError(f"could not sample {target_size} unique learners; got {len(chosen_rows)}")
    result = pd.DataFrame(chosen_rows).drop(columns=["_order", "stratum"], errors="ignore")
    result["panel"] = f"Panel {panel}"
    return result.reset_index(drop=True)


def validate_panels(panel_a: pd.DataFrame, panel_b: pd.DataFrame, state: pd.DataFrame) -> list[str]:
    errors: list[str] = []
    for name, panel in (("A", panel_a), ("B", panel_b)):
        if panel.case_id.duplicated().any(): errors.append(f"panel_{name}_duplicate_case")
        if panel.student_id.duplicated().any(): errors.append(f"panel_{name}_duplicate_student")
        if panel.enrollment_identity.duplicated().any(): errors.append(f"panel_{name}_duplicate_enrollment")
        if not panel.stage.isin(STAGES).all(): errors.append(f"panel_{name}_invalid_stage")
        if not panel.risk_probability.between(0, 1).all(): errors.append(f"panel_{name}_invalid_probability")
        if set(panel.case_id) - set(state.case_id): errors.append(f"panel_{name}_case_not_in_state")
    if set(panel_a.case_id) & set(panel_b.case_id): errors.append("case_overlap")
    if set(panel_a.student_id) & set(panel_b.student_id): errors.append("student_overlap")
    if set(panel_a.enrollment_identity) & set(panel_b.enrollment_identity): errors.append("enrollment_overlap")
    if "FINAL-100" in set(panel_a.stage) | set(panel_b.stage): errors.append("final_stage_present")
    return errors
