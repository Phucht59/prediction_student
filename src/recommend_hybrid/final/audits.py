"""Scientific audits required before an explainable V2 release."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

from .ranker import FEATURE_COLUMNS


FORBIDDEN_RANKER_FEATURES = frozenset(
    {
        "action_id",
        "final_result",
        "label_confidence",
        "label_conflict",
        "label_entropy",
        "ood_score",
        "current_action_head_prediction",
        "causal_ate",
        "causal_cate",
        "student_key",
        "student_id",
    }
)


def assert_ranker_schema(columns: Iterable[str]) -> None:
    observed = set(columns)
    forbidden = sorted(observed & FORBIDDEN_RANKER_FEATURES)
    if forbidden:
        raise ValueError(f"forbidden ranker features detected: {forbidden}")
    missing = sorted(set(FEATURE_COLUMNS) - observed)
    if missing:
        raise ValueError(f"required ranker features are missing: {missing}")


def assert_student_disjoint_splits(
    split_students: Mapping[str, Sequence[str]],
) -> None:
    """Fail when any real learner appears in more than one named split."""

    names = tuple(split_students)
    normalized = {name: set(map(str, split_students[name])) for name in names}
    for left_index, left in enumerate(names):
        for right in names[left_index + 1 :]:
            overlap = normalized[left] & normalized[right]
            if overlap:
                examples = sorted(overlap)[:10]
                raise ValueError(
                    f"student overlap between {left} and {right}: "
                    f"count={len(overlap)}, examples={examples}"
                )


def assert_pre_cutoff_lineage(
    lineage: pd.DataFrame,
    *,
    feature_column: str = "feature_name",
    observation_end_column: str = "observation_end_day",
    cutoff_column: str = "cutoff_day",
    schedule_features: frozenset[str] = frozenset(
        {"assessments_due", "time_to_deadline_days", "quiz_available"}
    ),
) -> None:
    required = {feature_column, observation_end_column, cutoff_column}
    missing = required - set(lineage.columns)
    if missing:
        raise ValueError(f"lineage table is missing columns: {sorted(missing)}")
    behavioral = lineage.loc[~lineage[feature_column].isin(schedule_features)].copy()
    invalid = behavioral.loc[
        behavioral[observation_end_column].notna()
        & (behavioral[observation_end_column] >= behavioral[cutoff_column])
    ]
    if not invalid.empty:
        examples = invalid.head(10).to_dict(orient="records")
        raise ValueError(
            f"post-cutoff behavioral feature usage detected: count={len(invalid)}, "
            f"examples={examples}"
        )


def permute_context_by_query(
    frame: pd.DataFrame,
    *,
    query_column: str = "query_id",
    stage_column: str = "stage",
    feature_columns: Sequence[str] = FEATURE_COLUMNS,
    seed: int = 42,
) -> pd.DataFrame:
    """Permute whole learner contexts within stage while preserving query targets.

    Every action row belonging to one query receives the same donor context.
    Stage is preserved and not permuted. Action IDs, eligibility, relevance
    labels, and query membership are not modified. A valid contextual ranker
    must degrade under this audit.
    """

    required = {query_column, stage_column, *feature_columns}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"permutation frame is missing columns: {sorted(missing)}")

    permutable_columns = tuple(
        column for column in feature_columns if column != stage_column
    )
    context = frame.drop_duplicates(query_column).loc[
        :, [query_column, stage_column, *permutable_columns]
    ]
    if context[query_column].duplicated().any():
        raise ValueError("query context must be unique")

    rng = np.random.default_rng(seed)
    donor_rows: list[pd.DataFrame] = []
    for _, stage_context in context.groupby(stage_column, sort=False):
        donor = stage_context.copy()
        values = donor.loc[:, list(permutable_columns)].to_numpy(copy=True)
        donor.loc[:, list(permutable_columns)] = values[
            rng.permutation(len(values))
        ]
        donor_rows.append(donor)
    permuted_context = pd.concat(donor_rows, ignore_index=True)

    output = frame.drop(columns=list(permutable_columns)).merge(
        permuted_context.drop(columns=[stage_column]),
        on=query_column,
        how="left",
        validate="many_to_one",
    )
    return output.loc[:, frame.columns]


def context_permutation_degradation(
    original_metric: float,
    permuted_metric: float,
) -> float:
    if not np.isfinite(original_metric) or not np.isfinite(permuted_metric):
        raise ValueError("context permutation metrics must be finite")
    return float(original_metric - permuted_metric)


__all__ = [
    "FORBIDDEN_RANKER_FEATURES",
    "assert_pre_cutoff_lineage",
    "assert_ranker_schema",
    "assert_student_disjoint_splits",
    "context_permutation_degradation",
    "permute_context_by_query",
]
