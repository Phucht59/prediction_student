"""Memory-safe post/baseline OULAD VLE aggregation using temporary SQLite."""
from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Mapping

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
RAW = ROOT / "data/raw"
DEFAULT_DATABASE = ROOT / "artifacts/recommend_hybrid/causal/runtime/oulad_activity.sqlite"
CONTENT_TYPES = {
    "oucontent",
    "resource",
    "page",
    "url",
    "glossary",
    "homepage",
    "subpage",
    "dataplus",
}


def collect_weekly_activity_sqlite(
    windows: Mapping[str, pd.DataFrame],
    *,
    chunksize: int,
    database_path: Path = DEFAULT_DATABASE,
    keep_database: bool = False,
) -> dict[str, pd.DataFrame]:
    """Aggregate all four landmark windows without retaining chunk outputs in RAM."""

    if chunksize <= 0:
        raise ValueError("chunksize must be positive")
    database_path.parent.mkdir(parents=True, exist_ok=True)
    if database_path.exists():
        database_path.unlink()
    site = pd.read_csv(
        RAW / "vle.csv",
        usecols=["code_module", "code_presentation", "id_site", "activity_type"],
    ).drop_duplicates()
    keys = ["code_module", "code_presentation", "id_student"]
    merge_windows = {
        stage: frame.assign(id_student=frame["student_id"].astype(int)).loc[
            :,
            [
                *keys,
                "record_id",
                "baseline_start_day",
                "cutoff_day",
                "followup_end_day",
            ],
        ]
        for stage, frame in windows.items()
    }
    usecols = [
        "code_module",
        "code_presentation",
        "id_student",
        "id_site",
        "date",
        "sum_click",
    ]
    connection = sqlite3.connect(database_path)
    try:
        connection.execute("PRAGMA journal_mode=OFF")
        connection.execute("PRAGMA synchronous=OFF")
        connection.execute("PRAGMA temp_store=FILE")
        connection.execute("PRAGMA cache_size=-131072")
        connection.execute(
            """
            CREATE TABLE daily_activity (
                stage TEXT NOT NULL,
                record_id TEXT NOT NULL,
                period TEXT NOT NULL,
                relative_week INTEGER NOT NULL,
                activity_date INTEGER NOT NULL,
                total_clicks REAL NOT NULL,
                quiz_clicks REAL NOT NULL,
                content_clicks REAL NOT NULL
            )
            """
        )
        for chunk in pd.read_csv(
            RAW / "studentVle.csv",
            usecols=usecols,
            chunksize=chunksize,
        ):
            chunk = chunk.merge(
                site,
                on=["code_module", "code_presentation", "id_site"],
                how="left",
                validate="many_to_one",
            )
            for stage, window in merge_windows.items():
                selected = chunk.merge(
                    window,
                    on=keys,
                    how="inner",
                    validate="many_to_one",
                )
                selected = selected.loc[
                    (selected["date"] >= selected["baseline_start_day"])
                    & (selected["date"] < selected["followup_end_day"])
                ].copy()
                if selected.empty:
                    continue
                baseline = selected["date"] < selected["cutoff_day"]
                selected["period"] = np.where(baseline, "baseline", "followup")
                selected["period_start"] = np.where(
                    baseline,
                    selected["baseline_start_day"],
                    selected["cutoff_day"],
                )
                selected["relative_week"] = (
                    (selected["date"] - selected["period_start"]) // 7
                ).astype(int)
                selected["quiz_clicks"] = np.where(
                    selected["activity_type"].eq("quiz"),
                    selected["sum_click"],
                    0,
                )
                selected["content_clicks"] = np.where(
                    selected["activity_type"].isin(CONTENT_TYPES),
                    selected["sum_click"],
                    0,
                )
                daily = selected.groupby(
                    ["record_id", "period", "relative_week", "date"],
                    as_index=False,
                    sort=False,
                ).agg(
                    total_clicks=("sum_click", "sum"),
                    quiz_clicks=("quiz_clicks", "sum"),
                    content_clicks=("content_clicks", "sum"),
                )
                daily.insert(0, "stage", stage)
                daily = daily.rename(columns={"date": "activity_date"})
                # The default executemany path avoids SQLite's platform-specific
                # maximum bind-variable limit triggered by method="multi".
                daily.to_sql(
                    "daily_activity",
                    connection,
                    if_exists="append",
                    index=False,
                    chunksize=10_000,
                )
            connection.commit()
        connection.execute(
            "CREATE INDEX idx_daily_stage_record ON daily_activity(stage, record_id, period, relative_week)"
        )
        connection.commit()
        weekly = pd.read_sql_query(
            """
            SELECT
                stage,
                record_id,
                period,
                relative_week,
                SUM(total_clicks) AS total_clicks,
                SUM(quiz_clicks) AS quiz_clicks,
                SUM(content_clicks) AS content_clicks,
                COUNT(DISTINCT activity_date) AS active_days
            FROM daily_activity
            GROUP BY stage, record_id, period, relative_week
            ORDER BY stage, record_id, period, relative_week
            """,
            connection,
        )
    finally:
        connection.close()
        if database_path.exists() and not keep_database:
            database_path.unlink()
    columns = [
        "record_id",
        "period",
        "relative_week",
        "total_clicks",
        "quiz_clicks",
        "content_clicks",
        "active_days",
    ]
    return {
        stage: weekly.loc[weekly["stage"].eq(stage), columns].reset_index(drop=True)
        for stage in windows
    }


__all__ = ["DEFAULT_DATABASE", "collect_weekly_activity_sqlite"]
