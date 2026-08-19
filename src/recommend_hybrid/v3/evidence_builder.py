"""Cutoff-safe OULAD recommendation evidence. observation_start <= t < cutoff."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .contracts import STAGE_FRACTION, Stage

QUIZ_TYPES = frozenset({"quiz", "externalquiz"})
CONTENT_TYPES = frozenset(
    {
        "oucontent",
        "resource",
        "page",
        "subpage",
        "sharedsubpage",
        "url",
        "folder",
        "glossary",
        "dataplus",
        "dualpane",
        "ouwiki",
    }
)
FORBIDDEN = frozenset({"final_result", "target", "date_unregistration", "score"})


def _assert_no_forbidden(columns) -> None:
    leaked = [c for c in columns if str(c).lower() in FORBIDDEN]
    if leaked:
        raise ValueError(f"forbidden columns in evidence table: {leaked}")


def assessment_evidence(queries: pd.DataFrame, raw_dir: Path) -> pd.DataFrame:
    definitions = pd.read_csv(raw_dir / "assessments.csv", usecols=["code_module", "code_presentation", "id_assessment", "date"])
    definitions = definitions.loc[definitions["date"].notna()].copy()
    definitions["date"] = pd.to_numeric(definitions["date"], errors="coerce")
    definitions = definitions.dropna(subset=["date"])
    submissions = pd.read_csv(raw_dir / "studentAssessment.csv", usecols=["id_assessment", "id_student", "date_submitted"])
    submissions["date_submitted"] = pd.to_numeric(submissions["date_submitted"], errors="coerce")
    submissions = submissions.dropna(subset=["date_submitted"]).groupby(["id_student", "id_assessment"], as_index=False)["date_submitted"].min()
    expanded = queries[["query_id", "id_student", "code_module", "code_presentation", "cutoff_day"]].merge(
        definitions, on=["code_module", "code_presentation"], how="left"
    )
    expanded = expanded.merge(submissions, on=["id_student", "id_assessment"], how="left")
    due_before = expanded["date"].notna() & (expanded["date"] < expanded["cutoff_day"])
    completed = due_before & expanded["date_submitted"].notna() & (expanded["date_submitted"] < expanded["cutoff_day"])
    due_soon = expanded["date"].notna() & (expanded["date"] >= expanded["cutoff_day"]) & (expanded["date"] < expanded["cutoff_day"] + 14)
    remaining = expanded["date"].notna() & (expanded["date"] >= expanded["cutoff_day"])
    expanded["due_before"] = due_before
    expanded["completed"] = completed
    expanded["due_soon"] = due_soon
    expanded["remaining"] = remaining
    expanded["days_to_due"] = np.where(remaining, expanded["date"] - expanded["cutoff_day"], np.nan)
    metrics = expanded.groupby("query_id", as_index=False).agg(
        assessments_due=("due_before", "sum"),
        assessments_completed=("completed", "sum"),
        due_soon_count=("due_soon", "sum"),
        remaining_count=("remaining", "sum"),
        time_to_deadline_days=("days_to_due", "min"),
    )
    metrics["missing_assessment_count"] = (metrics["assessments_due"] - metrics["assessments_completed"]).astype(int)
    metrics["completion_rate"] = np.where(metrics["assessments_due"] > 0, metrics["assessments_completed"] / metrics["assessments_due"], np.nan)
    metrics["assessment_progress"] = metrics["completion_rate"]
    metrics["assessment_window_open"] = metrics["remaining_count"] > 0
    metrics["knowledge_gap_evidence"] = (metrics["missing_assessment_count"] > 0) | (
        metrics["completion_rate"].notna() & (metrics["completion_rate"] < 0.8)
    )
    return metrics


def vle_availability(queries: pd.DataFrame, raw_dir: Path) -> pd.DataFrame:
    vle = pd.read_csv(raw_dir / "vle.csv", usecols=["code_module", "code_presentation", "activity_type", "week_from"])
    vle["activity_type"] = vle["activity_type"].astype(str).str.lower()
    vle["week_from_num"] = pd.to_numeric(vle["week_from"], errors="coerce")
    grouped = {(str(m), str(p)): frame for (m, p), frame in vle.groupby(["code_module", "code_presentation"], sort=False)}
    rows = []
    for row in queries[["query_id", "code_module", "code_presentation", "cutoff_day"]].itertuples(index=False):
        sites = grouped.get((str(row.code_module), str(row.code_presentation)))
        if sites is None:
            rows.append({"query_id": row.query_id, "vle_access_available": False, "study_material_available": False, "quiz_available": False})
            continue
        available = sites.loc[sites["week_from_num"].isna() | ((sites["week_from_num"] * 7) < int(row.cutoff_day))]
        types = set(available["activity_type"].tolist()) if not available.empty else set()
        rows.append(
            {
                "query_id": row.query_id,
                "vle_access_available": bool(types),
                "study_material_available": bool(types & CONTENT_TYPES),
                "quiz_available": bool(types & QUIZ_TYPES),
            }
        )
    return pd.DataFrame(rows)


def vle_behavior(queries: pd.DataFrame, raw_dir: Path) -> pd.DataFrame:
    wanted = set(queries["id_student"].astype(int))
    max_cutoff = int(queries["cutoff_day"].max())
    vle_map = pd.read_csv(raw_dir / "vle.csv", usecols=["id_site", "code_module", "code_presentation", "activity_type"])
    vle_map["activity_type"] = vle_map["activity_type"].astype(str).str.lower()
    type_lookup = {
        (str(r.code_module), str(r.code_presentation), int(r.id_site)): r.activity_type
        for r in vle_map.itertuples(index=False)
    }
    events: list[pd.DataFrame] = []
    for chunk in pd.read_csv(
        raw_dir / "studentVle.csv",
        usecols=["code_module", "code_presentation", "id_student", "id_site", "date", "sum_click"],
        chunksize=1_000_000,
    ):
        part = chunk.loc[chunk["id_student"].isin(wanted) & chunk["date"].notna() & (chunk["date"] >= 0) & (chunk["date"] < max_cutoff)].copy()
        if part.empty:
            continue
        events.append(part)
    if events:
        clicks = pd.concat(events, ignore_index=True)
    else:
        clicks = pd.DataFrame(columns=["code_module", "code_presentation", "id_student", "id_site", "date", "sum_click"])
    rows = []
    grouped = {key: frame for key, frame in clicks.groupby(["id_student", "code_module", "code_presentation"], sort=False)} if not clicks.empty else {}
    for row in queries.itertuples(index=False):
        key = (int(row.id_student), str(row.code_module), str(row.code_presentation))
        cutoff = int(row.cutoff_day)
        start = 0
        frame = grouped.get(key)
        if frame is None:
            rows.append(
                {
                    "query_id": row.query_id,
                    "inactivity_streak": None,
                    "active_day_rate": 0.0,
                    "recent_activity_trend": 0.0,
                    "regularity_score": 0.0,
                    "content_coverage": 0.0,
                    "quiz_activity": 0.0,
                }
            )
            continue
        kept = frame.loc[(frame["date"] >= start) & (frame["date"] < cutoff)].copy()
        observed_days = max(1, cutoff - start)
        if kept.empty:
            rows.append(
                {
                    "query_id": row.query_id,
                    "inactivity_streak": cutoff - start,
                    "active_day_rate": 0.0,
                    "recent_activity_trend": 0.0,
                    "regularity_score": 0.0,
                    "content_coverage": 0.0,
                    "quiz_activity": 0.0,
                }
            )
            continue
        last_day = int(kept["date"].max())
        streak = max(0, cutoff - 1 - last_day)
        active_days = int(kept["date"].nunique())
        active_day_rate = active_days / observed_days
        kept["week"] = (kept["date"] // 7).astype(int)
        n_weeks = max(1, int(np.ceil(cutoff / 7)))
        weekly = kept.groupby("week")["sum_click"].sum()
        flags = np.zeros(n_weeks, dtype=np.float32)
        for week, value in weekly.items():
            if 0 <= int(week) < n_weeks and value > 0:
                flags[int(week)] = 1.0
        regularity = float(1.0 - min(1.0, 2.0 * float(flags.std())))
        types = [
            type_lookup.get((str(row.code_module), str(row.code_presentation), int(site)), "other")
            for site in kept["id_site"].tolist()
        ]
        kept = kept.assign(activity_type=types)
        content_weeks = kept.loc[kept.activity_type.isin(CONTENT_TYPES), "week"].nunique()
        quiz_weeks = kept.loc[kept.activity_type.isin(QUIZ_TYPES), "week"].nunique()
        mid = cutoff / 2.0
        early = float(kept.loc[kept["date"] < mid, "sum_click"].sum())
        late = float(kept.loc[kept["date"] >= mid, "sum_click"].sum())
        trend = float(np.tanh(np.log1p(late) - np.log1p(early)))
        rows.append(
            {
                "query_id": row.query_id,
                "inactivity_streak": streak,
                "active_day_rate": float(active_day_rate),
                "recent_activity_trend": trend,
                "regularity_score": regularity,
                "content_coverage": float(content_weeks / n_weeks),
                "quiz_activity": float(quiz_weeks / n_weeks),
            }
        )
    return pd.DataFrame(rows)


def build_evidence_table(queries: pd.DataFrame, raw_dir: str | Path) -> pd.DataFrame:
    raw_dir = Path(raw_dir)
    required = {"query_id", "id_student", "code_module", "code_presentation", "cutoff_day", "stage"}
    missing = required - set(queries.columns)
    if missing:
        raise ValueError(f"queries missing {sorted(missing)}")
    _assert_no_forbidden(queries.columns)
    if queries["query_id"].duplicated().any():
        raise ValueError("duplicate query_id")
    table = queries.copy()
    table = table.merge(assessment_evidence(table, raw_dir), on="query_id", how="left")
    table = table.merge(vle_availability(table, raw_dir), on="query_id", how="left")
    table = table.merge(vle_behavior(table, raw_dir), on="query_id", how="left")
    if "course_progress" not in table.columns:
        table["course_progress"] = table["stage"].map(lambda s: STAGE_FRACTION[Stage(s)] if not isinstance(s, Stage) else STAGE_FRACTION[s])
    _assert_no_forbidden(table.columns)
    return table
