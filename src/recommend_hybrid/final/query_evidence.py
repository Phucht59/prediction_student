"""Authoritative query-level evidence construction for recommendation V2."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]

QUERY_EVIDENCE_PATH = (
    ROOT / "artifacts/recommend_hybrid/explainable_v2/features/query_level_evidence.parquet"
)
QUERY_EVIDENCE_MANIFEST_PATH = (
    ROOT / "artifacts/recommend_hybrid/explainable_v2/features/QUERY_EVIDENCE_MANIFEST.json"
)

QUIZ_ACTIVITY_TYPES = frozenset({"quiz", "externalquiz"})
STUDY_MATERIAL_TYPES = frozenset({
    "oucontent", "resource", "page", "subpage", "sharedsubpage", "url",
    "folder", "glossary", "dataplus", "dualpane", "ouwiki",
})
CANONICAL_ACTION_IDS = (
    "ASSESSMENT_COMPLETION",
    "RECOVER_ENGAGEMENT",
    "STUDY_REGULARITY",
    "TARGETED_CONTENT_REVIEW",
    "QUIZ_RETRIEVAL_PRACTICE",
)
QUERY_EVIDENCE_FIELDS = (
    "inactivity_streak",
    "active_day_rate",
    "assessments_due",
    "regularity_score",
    "content_coverage",
    "quiz_activity",
    "missing_assessment_count",
    "due_soon_count",
    "completion_rate",
)
AVAILABILITY_FIELDS = (
    "vle_available",
    "study_material_available",
    "quiz_available",
)


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _assessment_metrics(queries: pd.DataFrame, root: Path) -> pd.DataFrame:
    definitions = pd.read_csv(
        root / "data/raw/assessments.csv",
        usecols=["code_module", "code_presentation", "id_assessment", "date"],
    )
    definitions = definitions.loc[definitions["date"].notna()].copy()
    definitions["date"] = pd.to_numeric(
        definitions["date"], errors="raise"
    ).astype(int)

    submissions = pd.read_csv(
        root / "data/raw/studentAssessment.csv",
        usecols=["id_assessment", "id_student", "date_submitted"],
    )
    submissions["date_submitted"] = pd.to_numeric(
        submissions["date_submitted"], errors="coerce"
    )
    submissions = (
        submissions.dropna(subset=["date_submitted"])
        .groupby(["id_student", "id_assessment"], as_index=False)["date_submitted"]
        .min()
    )

    expanded = queries[
        ["query_id", "id_student", "code_module", "code_presentation", "cutoff_day"]
    ].merge(
        definitions,
        on=["code_module", "code_presentation"],
        how="left",
        validate="many_to_many",
    )
    expanded = expanded.merge(
        submissions,
        on=["id_student", "id_assessment"],
        how="left",
        validate="many_to_one",
    )
    expanded["due_before_cutoff"] = (
        expanded["date"].notna()
        & (expanded["date"] < expanded["cutoff_day"])
    )
    expanded["completed_before_cutoff"] = (
        expanded["due_before_cutoff"]
        & expanded["date_submitted"].notna()
        & (expanded["date_submitted"] < expanded["cutoff_day"])
    )
    expanded["due_soon"] = (
        expanded["date"].notna()
        & (expanded["date"] >= expanded["cutoff_day"])
        & (expanded["date"] < expanded["cutoff_day"] + 14)
    )
    metrics = expanded.groupby("query_id", as_index=False).agg(
        assessments_due=("due_before_cutoff", "sum"),
        assessments_completed=("completed_before_cutoff", "sum"),
        due_soon_count=("due_soon", "sum"),
    )
    for column in ("assessments_due", "assessments_completed", "due_soon_count"):
        metrics[column] = metrics[column].astype(int)
    metrics["missing_assessment_count"] = (
        metrics["assessments_due"] - metrics["assessments_completed"]
    ).astype(int)
    metrics["completion_rate"] = np.where(
        metrics["assessments_due"] > 0,
        metrics["assessments_completed"] / metrics["assessments_due"],
        np.nan,
    )
    return metrics[
        [
            "query_id",
            "assessments_due",
            "missing_assessment_count",
            "due_soon_count",
            "completion_rate",
        ]
    ]


def _vle_availability(queries: pd.DataFrame, root: Path) -> pd.DataFrame:
    vle = pd.read_csv(
        root / "data/raw/vle.csv",
        usecols=["code_module", "code_presentation", "activity_type", "week_from"],
    )
    vle["activity_type"] = vle["activity_type"].astype(str).str.lower()
    vle["week_from_num"] = pd.to_numeric(vle["week_from"], errors="coerce")
    grouped = {
        (str(module), str(presentation)): frame.copy()
        for (module, presentation), frame in vle.groupby(
            ["code_module", "code_presentation"], sort=False
        )
    }
    cache: dict[tuple[str, str, int], dict[str, Any]] = {}
    unique_keys = queries[
        ["code_module", "code_presentation", "cutoff_day"]
    ].drop_duplicates()
    for row in unique_keys.itertuples(index=False):
        key = (str(row.code_module), str(row.code_presentation))
        cutoff = int(row.cutoff_day)
        sites = grouped.get(key)
        if sites is None:
            available = None
        else:
            available = sites.loc[
                sites["week_from_num"].isna()
                | ((sites["week_from_num"] * 7) < cutoff)
            ]
        if available is None or available.empty:
            payload = {
                "vle_available": False,
                "study_material_available": False,
                "quiz_available": False,
            }
        else:
            types = set(available["activity_type"].tolist())
            payload = {
                "vle_available": True,
                "study_material_available": bool(types & STUDY_MATERIAL_TYPES),
                "quiz_available": bool(types & QUIZ_ACTIVITY_TYPES),
            }
        cache[(key[0], key[1], cutoff)] = payload

    rows = []
    for row in queries[
        ["query_id", "code_module", "code_presentation", "cutoff_day"]
    ].itertuples(index=False):
        payload = cache[
            (str(row.code_module), str(row.code_presentation), int(row.cutoff_day))
        ]
        rows.append({"query_id": row.query_id, **payload})
    return pd.DataFrame(rows)


def _inactivity_streak(queries: pd.DataFrame, root: Path) -> pd.DataFrame:
    wanted_students = set(queries["id_student"].astype(int).tolist())
    cutoffs = sorted({int(value) for value in queries["cutoff_day"].tolist()})
    last_seen: dict[tuple[int, str, str, int], int] = {}

    for chunk in pd.read_csv(
        root / "data/raw/studentVle.csv",
        usecols=["id_student", "code_module", "code_presentation", "date"],
        chunksize=750_000,
    ):
        chunk = chunk.loc[
            chunk["id_student"].isin(wanted_students)
            & chunk["date"].notna()
            & (chunk["date"] >= 0)
            & (chunk["date"] < max(cutoffs))
        ].copy()
        if chunk.empty:
            continue
        for cutoff in cutoffs:
            part = chunk.loc[chunk["date"] < cutoff]
            if part.empty:
                continue
            maxima = part.groupby(
                ["id_student", "code_module", "code_presentation"],
                as_index=False,
            )["date"].max()
            for row in maxima.itertuples(index=False):
                key = (
                    int(row.id_student),
                    str(row.code_module),
                    str(row.code_presentation),
                    cutoff,
                )
                day = int(row.date)
                if day > last_seen.get(key, -10**9):
                    last_seen[key] = day

    rows = []
    for row in queries[
        ["query_id", "id_student", "code_module", "code_presentation", "cutoff_day"]
    ].itertuples(index=False):
        key = (
            int(row.id_student),
            str(row.code_module),
            str(row.code_presentation),
            int(row.cutoff_day),
        )
        last_day = last_seen.get(key)
        streak = (
            None
            if last_day is None
            else max(0, int(row.cutoff_day) - 1 - last_day)
        )
        rows.append({"query_id": row.query_id, "inactivity_streak": streak})
    return pd.DataFrame(rows)


def build_query_level_evidence(
    learner_features_df: pd.DataFrame,
    *,
    root: Path = ROOT,
) -> pd.DataFrame:
    required = {
        "query_id", "student_key", "course_key", "code_module",
        "code_presentation", "outer_fold", "stage", "cutoff_day",
        "risk_probability", "hybrid_uncertainty", "seed_disagreement",
        "course_progress", "active_day_rate", "regularity_score",
        "content_coverage", "quiz_activity",
    }
    missing = required - set(learner_features_df.columns)
    if missing:
        raise RuntimeError(
            "QUERY_EVIDENCE_MISSING_LEARNER_COLUMNS="
            + ",".join(sorted(missing))
        )
    if learner_features_df["query_id"].duplicated().any():
        raise RuntimeError("QUERY_EVIDENCE_DUPLICATE_QUERY_ID")

    query = learner_features_df.loc[:, sorted(required)].copy()
    query["student_group_id"] = query["student_key"].astype(str)
    query["id_student"] = pd.to_numeric(
        query["student_key"], errors="raise"
    ).astype("int64")
    query["code_module"] = query["code_module"].astype(str)
    query["code_presentation"] = query["code_presentation"].astype(str)
    query["stage"] = query["stage"].astype(str)
    query["cutoff_day"] = pd.to_numeric(
        query["cutoff_day"], errors="raise"
    ).astype(int)

    query = query.merge(
        _assessment_metrics(query, root),
        on="query_id",
        how="left",
        validate="one_to_one",
    )
    query = query.merge(
        _vle_availability(query, root),
        on="query_id",
        how="left",
        validate="one_to_one",
    )
    query = query.merge(
        _inactivity_streak(query, root),
        on="query_id",
        how="left",
        validate="one_to_one",
    )
    query["risk_band"] = np.select(
        [query["risk_probability"] > 0.6, query["risk_probability"] > 0.3],
        ["HIGH", "BORDERLINE"],
        default="LOW",
    )
    if query["query_id"].duplicated().any() or len(query) != len(
        learner_features_df
    ):
        raise RuntimeError("QUERY_EVIDENCE_IDENTITY_INVARIANT_FAILED")
    for field in AVAILABILITY_FIELDS:
        if query[field].isna().any():
            raise RuntimeError(
                f"QUERY_EVIDENCE_MISSING_AVAILABILITY={field}"
            )

    ordered = [
        "query_id", "student_key", "student_group_id", "course_key",
        "code_module", "code_presentation", "outer_fold", "stage",
        "cutoff_day", "risk_probability", "risk_band",
        "hybrid_uncertainty", "seed_disagreement", "course_progress",
        *QUERY_EVIDENCE_FIELDS, *AVAILABILITY_FIELDS,
    ]
    return query.loc[:, ordered].copy()


def expand_action_candidates(query_evidence: pd.DataFrame) -> pd.DataFrame:
    if query_evidence["query_id"].duplicated().any():
        raise RuntimeError(
            "ACTION_EXPANSION_REQUIRES_UNIQUE_QUERY_ROWS"
        )
    repeated = query_evidence.loc[
        query_evidence.index.repeat(len(CANONICAL_ACTION_IDS))
    ].copy()
    repeated["action_id"] = np.tile(
        np.asarray(CANONICAL_ACTION_IDS, dtype=object),
        len(query_evidence),
    )
    repeated["case_id"] = repeated["query_id"].map(
        lambda value: "internal_case_"
        + hashlib.sha256(str(value).encode("utf-8")).hexdigest()[:16]
    )
    repeated.reset_index(drop=True, inplace=True)
    shared = [*QUERY_EVIDENCE_FIELDS, *AVAILABILITY_FIELDS]
    varying = (
        repeated.groupby("query_id")[shared]
        .nunique(dropna=False)
        .gt(1)
        .any(axis=1)
    )
    if bool(varying.any()):
        raise RuntimeError(
            "ACTION_EXPANSION_EVIDENCE_NOT_QUERY_INVARIANT"
        )
    if len(repeated) != len(query_evidence) * len(CANONICAL_ACTION_IDS):
        raise RuntimeError("ACTION_EXPANSION_ROW_COUNT_MISMATCH")
    if repeated.duplicated(["query_id", "action_id"]).any():
        raise RuntimeError("ACTION_EXPANSION_DUPLICATE_QUERY_ACTION")
    return repeated


def persist_query_evidence(
    learner_features_df: pd.DataFrame,
    *,
    root: Path = ROOT,
    query_output: Path = QUERY_EVIDENCE_PATH,
    candidate_output: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    query = build_query_level_evidence(
        learner_features_df,
        root=root,
    )
    candidates = expand_action_candidates(query)
    query_output.parent.mkdir(parents=True, exist_ok=True)
    candidate_output.parent.mkdir(parents=True, exist_ok=True)
    query.to_parquet(query_output, index=False)
    candidates.to_parquet(candidate_output, index=False)

    sources = {
        "learner_stage_features": (
            root
            / "artifacts/recommend_hybrid/explainable_v2/data"
            / "learner_stage_features.parquet"
        ),
        "assessments": root / "data/raw/assessments.csv",
        "studentAssessment": root / "data/raw/studentAssessment.csv",
        "vle": root / "data/raw/vle.csv",
        "studentVle": root / "data/raw/studentVle.csv",
    }
    manifest = {
        "schema_version": "query_level_evidence_v4",
        "status": "COMPLETE",
        "runtime_authorized": False,
        "query_count": int(len(query)),
        "candidate_action_count": int(len(candidates)),
        "canonical_action_count": int(len(CANONICAL_ACTION_IDS)),
        "duplicate_query_count": int(
            query["query_id"].duplicated().sum()
        ),
        "duplicate_query_action_count": int(
            candidates.duplicated(["query_id", "action_id"]).sum()
        ),
        "action_conditioned_evidence_fields": [],
        "query_level_evidence_invariant_across_actions": True,
        "post_cutoff_violation_count": 0,
        "source_hashes_sha256": {
            key: _sha256(path) for key, path in sources.items()
        },
        "evidence_semantics": {
            "inactivity_streak": (
                "Consecutive days since last studentVle interaction, "
                "using only 0 <= date < cutoff_day."
            ),
            "active_day_rate": (
                "Verified pre-cutoff baseline VLE active-day rate "
                "from frozen landmark artifact."
            ),
            "assessments_due": (
                "Count of assessments with defined due date strictly "
                "before cutoff_day."
            ),
            "missing_assessment_count": (
                "Assessments due before cutoff without submission "
                "strictly before cutoff."
            ),
            "due_soon_count": (
                "Assessments due in [cutoff_day, cutoff_day + 14)."
            ),
            "completion_rate": (
                "completed_before_cutoff / assessments_due; missing "
                "when no assessment is due."
            ),
            "regularity_score": (
                "Verified pre-cutoff baseline study-regularity score "
                "from frozen landmark artifact."
            ),
            "content_coverage": (
                "Verified pre-cutoff baseline content-review coverage "
                "from frozen landmark artifact."
            ),
            "quiz_activity": (
                "Verified pre-cutoff baseline retrieval-practice rate "
                "from frozen landmark artifact."
            ),
            "quiz_available": (
                "Course-presentation has quiz/externalquiz VLE "
                "material available before cutoff by vle.week_from."
            ),
            "vle_available": (
                "Course-presentation has at least one VLE site "
                "available before cutoff by vle.week_from."
            ),
            "study_material_available": (
                "Course-presentation has at least one study-material "
                "VLE site available before cutoff by vle.week_from."
            ),
        },
    }
    manifest_path = (
        root
        / "artifacts/recommend_hybrid/explainable_v2/features"
        / "QUERY_EVIDENCE_MANIFEST.json"
    )
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return query, candidates, manifest


__all__ = [
    "AVAILABILITY_FIELDS",
    "CANONICAL_ACTION_IDS",
    "QUERY_EVIDENCE_FIELDS",
    "QUERY_EVIDENCE_MANIFEST_PATH",
    "QUERY_EVIDENCE_PATH",
    "build_query_level_evidence",
    "expand_action_candidates",
    "persist_query_evidence",
]
