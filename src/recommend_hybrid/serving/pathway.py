"""Remaining-assessment pathway Q_τ. Deadlines strictly after cutoff."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from .contracts import PathwayItem


def load_assessment_tables(raw_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    definitions = pd.read_csv(
        raw_dir / "assessments.csv",
        usecols=["code_module", "code_presentation", "id_assessment", "date", "assessment_type"],
    )
    definitions["date"] = pd.to_numeric(definitions["date"], errors="coerce")
    definitions = definitions.dropna(subset=["date"]).copy()
    submissions = pd.read_csv(
        raw_dir / "studentAssessment.csv",
        usecols=["id_student", "id_assessment", "date_submitted", "score"],
    )
    submissions["date_submitted"] = pd.to_numeric(submissions["date_submitted"], errors="coerce")
    submissions["id_student"] = submissions["id_student"].astype(str)
    return definitions, submissions


def pathway_for_row(
    row: pd.Series | dict,
    definitions: pd.DataFrame,
    submissions: pd.DataFrame,
    *,
    limit: int = 3,
) -> tuple[PathwayItem, ...]:
    module = str(row["code_module"] if "code_module" in row else row.get("code_module"))
    presentation = str(row["code_presentation"] if "code_presentation" in row else row.get("code_presentation"))
    student = str(row["id_student"])
    cutoff = int(row["cutoff_day"])
    due = definitions.loc[
        (definitions["code_module"].astype(str) == module)
        & (definitions["code_presentation"].astype(str) == presentation)
        & (definitions["date"] > cutoff)
    ].copy()
    if due.empty:
        return ()
    submitted = submissions.loc[
        (submissions["id_student"] == student)
        & submissions["date_submitted"].notna()
        & (submissions["date_submitted"] <= cutoff)
    ]
    done = set(submitted["id_assessment"].tolist())
    due = due.loc[~due["id_assessment"].isin(done)].sort_values("date")
    items: list[PathwayItem] = []
    for rec in due.head(limit).itertuples(index=False):
        deadline = int(rec.date)
        items.append(
            PathwayItem(
                assessment_id=int(rec.id_assessment),
                deadline_day=deadline,
                days_until_due=deadline - cutoff,
            )
        )
    return tuple(items)


__all__ = ["load_assessment_tables", "pathway_for_row"]
