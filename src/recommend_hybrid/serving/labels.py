"""14-day persistence labels from OULAD logs. Outcomes are never features."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .contracts import PERSIST_WINDOW_DAYS, PersistLabel


def _assessment_persist(
    queries: pd.DataFrame,
    raw_dir: Path,
    window: int = PERSIST_WINDOW_DAYS,
) -> pd.Series:
    definitions = pd.read_csv(
        raw_dir / "assessments.csv",
        usecols=["code_module", "code_presentation", "id_assessment", "date"],
    )
    definitions["date"] = pd.to_numeric(definitions["date"], errors="coerce")
    definitions = definitions.dropna(subset=["date"])
    submissions = pd.read_csv(
        raw_dir / "studentAssessment.csv",
        usecols=["id_student", "id_assessment", "date_submitted"],
    )
    submissions["date_submitted"] = pd.to_numeric(submissions["date_submitted"], errors="coerce")
    submissions["id_student"] = submissions["id_student"].astype(str)
    queries = queries.copy()
    queries["id_student"] = queries["id_student"].astype(str)
    expanded = queries[["query_id", "id_student", "code_module", "code_presentation", "cutoff_day"]].merge(
        definitions, on=["code_module", "code_presentation"], how="left"
    )
    expanded = expanded.merge(submissions, on=["id_student", "id_assessment"], how="left")
    due_before = expanded["date"].notna() & (expanded["date"] < expanded["cutoff_day"])
    submitted_before = expanded["date_submitted"].notna() & (expanded["date_submitted"] < expanded["cutoff_day"])
    missing_at_tau = due_before & ~submitted_before
    submitted_in_window = (
        expanded["date_submitted"].notna()
        & (expanded["date_submitted"] > expanded["cutoff_day"])
        & (expanded["date_submitted"] <= expanded["cutoff_day"] + window)
    )
    still_missing = missing_at_tau & ~submitted_in_window
    stuck = still_missing.groupby(expanded["query_id"]).any()
    resolved = (missing_at_tau & submitted_in_window).groupby(expanded["query_id"]).any()
    had_missing = missing_at_tau.groupby(expanded["query_id"]).any()
    frame = pd.DataFrame({"assess_stuck": stuck, "assess_resolved": resolved, "had_missing": had_missing})
    return frame.reindex(queries["query_id"]).fillna(False)


def _vle_return(
    queries: pd.DataFrame,
    raw_dir: Path,
    window: int = PERSIST_WINDOW_DAYS,
) -> pd.Series:
    wanted = set(queries["id_student"].astype(str))
    min_cut = int(queries["cutoff_day"].min())
    max_cut = int(queries["cutoff_day"].max())
    returned: dict[tuple[str, str, str], set[int]] = {}
    for chunk in pd.read_csv(
        raw_dir / "studentVle.csv",
        usecols=["code_module", "code_presentation", "id_student", "date", "sum_click"],
        chunksize=1_000_000,
        dtype={"id_student": str, "code_module": str, "code_presentation": str},
    ):
        part = chunk.loc[
            chunk["id_student"].isin(wanted)
            & chunk["date"].notna()
            & (chunk["sum_click"] > 0)
        ].copy()
        if part.empty:
            continue
        part["date"] = pd.to_numeric(part["date"], errors="coerce")
        part = part.dropna(subset=["date"])
        part = part.loc[(part["date"] >= min_cut) & (part["date"] <= max_cut + window)]
        if part.empty:
            continue
        for rec in part.itertuples(index=False):
            key = (str(rec.id_student), str(rec.code_module), str(rec.code_presentation))
            returned.setdefault(key, set()).add(int(rec.date))
    flags = []
    for rec in queries.itertuples(index=False):
        key = (str(rec.id_student), str(rec.code_module), str(rec.code_presentation))
        cutoff = int(rec.cutoff_day)
        days = returned.get(key, set())
        flags.append(any(cutoff < day <= cutoff + window for day in days))
    return pd.Series(flags, index=queries["query_id"].tolist(), name="vle_returned")


def attach_persistence_labels(
    queries: pd.DataFrame,
    raw_dir: Path,
    *,
    window: int = PERSIST_WINDOW_DAYS,
) -> pd.DataFrame:
    """Add persist_label and stuck/resolved flags from logs after the cutoff."""
    frame = queries.copy()
    if "query_id" not in frame.columns:
        raise ValueError("queries require query_id")
    assess = _assessment_persist(frame, raw_dir, window)
    vle_back = _vle_return(frame, raw_dir, window)
    frame = frame.set_index("query_id", drop=False)
    frame["assess_stuck"] = assess["assess_stuck"].astype(bool)
    frame["assess_resolved"] = assess["assess_resolved"].astype(bool)
    frame["had_missing"] = assess["had_missing"].astype(bool)
    frame["vle_returned"] = vle_back.reindex(frame.index).fillna(False).astype(bool)
    missing = frame["missing_assessment_count"].fillna(0).astype(int)
    due = frame["due_soon_count"].fillna(0).astype(int)
    assess_at_tau = (missing >= 1) | (due >= 2)
    vle_ok = frame["vle_access_available"].fillna(False).astype(bool)
    streak = frame["inactivity_streak"].fillna(0).astype(int)
    active = frame["active_day_rate"].fillna(1.0).astype(float)
    engage_at_tau = vle_ok & ((streak >= 7) | (active < 0.20))
    frame["engage_at_tau"] = engage_at_tau.to_numpy()
    frame["assess_at_tau"] = assess_at_tau.to_numpy()
    frame["engage_stuck"] = engage_at_tau.to_numpy() & ~frame["vle_returned"].to_numpy()
    frame["engage_resolved"] = engage_at_tau.to_numpy() & frame["vle_returned"].to_numpy()
    labels = np.full(len(frame), PersistLabel.COUNSEL.value, dtype=object)
    labels = np.where(frame["engage_stuck"].to_numpy(), PersistLabel.ENGAGE.value, labels)
    labels = np.where(frame["assess_stuck"].to_numpy(), PersistLabel.ASSESS.value, labels)
    frame["persist_label"] = labels
    frame["resolved_assigned"] = False
    return frame.reset_index(drop=True)


__all__ = ["attach_persistence_labels"]
