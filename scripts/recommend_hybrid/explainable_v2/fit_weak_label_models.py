"""Panel-A-only, outer-fold train-only Snorkel aggregation for recommendation V2."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.recommend_hybrid.explainable_v2.action_eligibility import (
    evaluate_action_eligibility,
)
from src.recommend_hybrid.explainable_v2.contracts import CanonicalAction
from src.recommend_hybrid.explainable_v2.weak_labels import (
    ABSTAIN,
    CARDINALITY,
    WeakLabelSource,
    fit_label_model,
    source_correlation_audit,
    validate_vote_matrix,
)

EXPECTED_FROZEN_SHA256 = (
    "4a4871426880bdcd1257dc15c29a36c23de34481f07be68d8e5095dc20efefb9"
)
EXPECTED_PANEL_A_CASES = 300
EXPECTED_FROZEN_RECORDS = 1117
EXPECTED_ACTION_ROWS = 1500
EXPECTED_CASE_LINEAGE_SHA256 = (
    "5a804a74da464df22fe3a569156af6fcd9110ecc3ee81ee1348c9310ea4fc137"
)
SEED_BASE = 2026
EPOCHS = 1000
MINIMUM_INDEPENDENT_SOURCE_FAMILIES = 2
RETAINED_LABEL_STATUS = "OOF_PANEL_A_SILVER_LABEL"
INSUFFICIENT_SUPPORT_LABEL_STATUS = "INSUFFICIENT_SOURCE_SUPPORT"

FROZEN_PATH = (
    ROOT
    / "artifacts/recommend_hybrid/explainable_v2/annotations/frozen/panel_a_v1"
    / "panel_a_external_reviews_frozen.jsonl"
)
CANDIDATES_PATH = (
    ROOT
    / "artifacts/recommend_hybrid/explainable_v2/features/action_candidates.parquet"
)
OUTPUT_DIR = ROOT / "artifacts/recommend_hybrid/explainable_v2/labels/panel_a_v1"

SOURCES = (
    WeakLabelSource("LF_ASSESSMENT_NEED_V4", "BEHAVIORAL"),
    WeakLabelSource("LF_ENGAGEMENT_RECOVERY_V4", "BEHAVIORAL"),
    WeakLabelSource("LF_STUDY_REGULARITY_V4", "BEHAVIORAL"),
    WeakLabelSource("LF_CONTENT_REVIEW_V4", "BEHAVIORAL"),
    WeakLabelSource("LF_QUIZ_RETRIEVAL_V4", "BEHAVIORAL"),
    WeakLabelSource("LF_FEASIBILITY_CONSTRAINT_V4", "FEASIBILITY"),
    WeakLabelSource("REAL_EXTERNAL_GEMINI_REVIEW_V4", "LLM_EXPERT"),
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _blinded_case_id(query_id: str, salt: str) -> str:
    return (
        "case_"
        + hashlib.sha256(
            salt.encode("utf-8") + b"_" + query_id.encode("utf-8")
        ).hexdigest()[:24]
    )


def _load_frozen() -> pd.DataFrame:
    if not FROZEN_PATH.is_file():
        raise RuntimeError(f"MISSING_FROZEN_PANEL_A={FROZEN_PATH}")
    actual_sha = _sha256(FROZEN_PATH)
    if actual_sha != EXPECTED_FROZEN_SHA256:
        raise RuntimeError(
            "FROZEN_PANEL_A_SHA_MISMATCH="
            f"{actual_sha} expected={EXPECTED_FROZEN_SHA256}"
        )
    rows = [
        json.loads(line)
        for line in FROZEN_PATH.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    frame = pd.DataFrame(rows)
    required = {
        "case_id",
        "panel_id",
        "action_id",
        "relevance_score",
        "abstain",
        "reviewer_type",
        "provider",
        "model_name",
        "model_version",
    }
    missing = required - set(frame.columns)
    if missing:
        raise RuntimeError(
            "FROZEN_PANEL_A_MISSING_FIELDS=" + ",".join(sorted(missing))
        )
    if len(frame) != EXPECTED_FROZEN_RECORDS:
        raise RuntimeError(
            f"FROZEN_PANEL_A_RECORD_COUNT={len(frame)} expected={EXPECTED_FROZEN_RECORDS}"
        )
    if frame["case_id"].nunique() != EXPECTED_PANEL_A_CASES:
        raise RuntimeError(
            f"FROZEN_PANEL_A_CASE_COUNT={frame['case_id'].nunique()} "
            f"expected={EXPECTED_PANEL_A_CASES}"
        )
    if set(frame["panel_id"].astype(str)) != {"PANEL_A"}:
        raise RuntimeError("NON_PANEL_A_RECORD_DETECTED")
    if set(frame["reviewer_type"].astype(str)) != {"REAL_EXTERNAL_LLM_REVIEW"}:
        raise RuntimeError("NON_REAL_EXTERNAL_REVIEW_DETECTED")
    if frame.duplicated(["case_id", "action_id"]).any():
        raise RuntimeError("DUPLICATE_FROZEN_CASE_ACTION")
    return frame


def _case_lineage_sha256(mapping: pd.DataFrame) -> str:
    canonical = (
        mapping[["query_id", "blinded_case_id"]]
        .astype(str)
        .drop_duplicates()
        .sort_values(["query_id", "blinded_case_id"])
        .rename(columns={"blinded_case_id": "case_id"})
    )
    payload = json.dumps(
        canonical.to_dict("records"),
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _load_case_mapping(candidates: pd.DataFrame) -> tuple[pd.DataFrame, str]:
    salt = os.environ.get("CASE_EXPORT_SALT", "").strip()
    if salt:
        qids = candidates["query_id"].astype(str).drop_duplicates()
        mapping = pd.DataFrame(
            {
                "query_id": qids,
                "blinded_case_id": [
                    _blinded_case_id(qid, salt)
                    for qid in qids
                ],
            }
        )
        source = "CASE_EXPORT_SALT_RECONSTRUCTION"
    else:
        if not (OUTPUT_DIR / "probabilistic_relevance_labels.parquet").is_file():
            raise RuntimeError(
                "CASE_LINEAGE_UNAVAILABLE: set CASE_EXPORT_SALT or preserve the "
                "verified prior Panel-A label artifact"
            )
        prior = pd.read_parquet(
            OUTPUT_DIR / "probabilistic_relevance_labels.parquet",
            columns=["query_id", "case_id"],
        )
        mapping = (
            prior.astype(str)
            .drop_duplicates()
            .rename(columns={"case_id": "blinded_case_id"})
        )
        source = "PINNED_PRIOR_LABEL_CASE_LINEAGE"

    actual_sha = _case_lineage_sha256(mapping)
    if actual_sha != EXPECTED_CASE_LINEAGE_SHA256:
        raise RuntimeError(
            f"CASE_LINEAGE_SHA_MISMATCH={actual_sha} "
            f"expected={EXPECTED_CASE_LINEAGE_SHA256}"
        )
    return mapping, source


def _select_panel_a_candidates(
    frozen: pd.DataFrame,
) -> tuple[pd.DataFrame, str]:
    if not CANDIDATES_PATH.is_file():
        raise RuntimeError(f"MISSING_ACTION_CANDIDATES={CANDIDATES_PATH}")

    candidates = pd.read_parquet(CANDIDATES_PATH)
    required = {
        "query_id",
        "outer_fold",
        "stage",
        "action_id",
        "assessments_due",
        "missing_assessment_count",
        "due_soon_count",
        "inactivity_streak",
        "regularity_score",
        "content_coverage",
        "quiz_activity",
        "vle_available",
        "study_material_available",
        "quiz_available",
    }
    missing = required - set(candidates.columns)
    if missing:
        raise RuntimeError(
            "ACTION_CANDIDATES_MISSING_FIELDS=" + ",".join(sorted(missing))
        )
    if candidates.duplicated(["query_id", "action_id"]).any():
        raise RuntimeError("DUPLICATE_QUERY_ACTION_IN_CANDIDATES")

    mapping, lineage_source = _load_case_mapping(candidates)
    frozen_cases = set(frozen["case_id"].astype(str))
    mapping = mapping[mapping["blinded_case_id"].isin(frozen_cases)].copy()

    if mapping["blinded_case_id"].nunique() != EXPECTED_PANEL_A_CASES:
        raise RuntimeError(
            "PANEL_A_LINEAGE_MATCH_COUNT="
            f"{mapping['blinded_case_id'].nunique()} expected={EXPECTED_PANEL_A_CASES}"
        )
    if mapping["blinded_case_id"].duplicated().any():
        raise RuntimeError("BLINDED_CASE_COLLISION_DETECTED")

    selected = candidates.merge(mapping, on="query_id", how="inner", validate="many_to_one")
    selected["case_id"] = selected["blinded_case_id"]
    selected.drop(columns=["blinded_case_id"], inplace=True)

    if len(selected) != EXPECTED_ACTION_ROWS:
        raise RuntimeError(
            f"PANEL_A_ACTION_ROWS={len(selected)} expected={EXPECTED_ACTION_ROWS}"
        )
    if selected["query_id"].nunique() != EXPECTED_PANEL_A_CASES:
        raise RuntimeError("PANEL_A_QUERY_COUNT_MISMATCH")
    if selected.groupby("query_id")["action_id"].nunique().ne(5).any():
        raise RuntimeError("PANEL_A_NOT_EXACTLY_FIVE_ACTIONS_PER_QUERY")
    return selected.reset_index(drop=True), lineage_source


def _is_missing(value) -> bool:
    return value is None or (isinstance(value, float) and np.isnan(value))


def _feasible(row: dict, action: str) -> bool:
    feature_row = {
        "inactivity_streak": row.get("inactivity_streak"),
        "active_day_rate": row.get("active_day_rate"),
        "assessments_due": row.get("assessments_due"),
        "regularity_score": row.get("regularity_score"),
        "content_coverage": row.get("content_coverage"),
        "quiz_activity": row.get("quiz_activity"),
        "missing_assessment_count": row.get("missing_assessment_count"),
        "due_soon_count": row.get("due_soon_count"),
        "completion_rate": row.get("completion_rate"),
        "vle_available": bool(row.get("vle_available", False)),
        "study_material_available": bool(row.get("study_material_available", False)),
        "quiz_available": bool(row.get("quiz_available", False)),
        "stage": str(row.get("stage")),
    }
    eligible, _ = evaluate_action_eligibility(feature_row, action)
    return bool(eligible)


def _review_lookup(frozen: pd.DataFrame) -> dict[tuple[str, str], int]:
    lookup: dict[tuple[str, str], int] = {}
    for row in frozen.to_dict("records"):
        if bool(row.get("abstain", False)):
            continue
        score = int(row["relevance_score"])
        if score not in (0, 1, 2, 3):
            raise RuntimeError(f"INVALID_EXTERNAL_RELEVANCE_SCORE={score}")
        lookup[(str(row["case_id"]), str(row["action_id"]))] = score
    return lookup


def evaluate_votes(
    candidates: pd.DataFrame,
    frozen: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray]:
    lookup = _review_lookup(frozen)
    L = np.full((len(candidates), len(SOURCES)), ABSTAIN, dtype=int)
    feasible = np.zeros(len(candidates), dtype=bool)

    for i, row in enumerate(candidates.to_dict("records")):
        action = str(row["action_id"])
        eligible = _feasible(row, action)
        feasible[i] = eligible

        if action == CanonicalAction.ASSESSMENT_COMPLETION.value:
            missing = row.get("missing_assessment_count")
            due_soon = row.get("due_soon_count")
            due = row.get("assessments_due")
            if not _is_missing(missing) and float(missing) > 0:
                L[i, 0] = 3
            elif not _is_missing(due_soon) and float(due_soon) > 0:
                L[i, 0] = 2
            elif not _is_missing(due) and float(due) > 0:
                L[i, 0] = 1

        if action == CanonicalAction.RECOVER_ENGAGEMENT.value:
            streak = row.get("inactivity_streak")
            if not _is_missing(streak):
                streak = float(streak)
                L[i, 1] = 3 if streak >= 7 else (2 if streak >= 3 else 1)

        if action == CanonicalAction.STUDY_REGULARITY.value:
            reg = row.get("regularity_score")
            if not _is_missing(reg):
                reg = float(reg)
                L[i, 2] = 3 if reg < 0.30 else (2 if reg < 0.50 else 1)

        if action == CanonicalAction.TARGETED_CONTENT_REVIEW.value:
            cov = row.get("content_coverage")
            if not _is_missing(cov):
                cov = float(cov)
                L[i, 3] = 3 if cov < 0.40 else (2 if cov < 0.65 else 1)

        if action == CanonicalAction.QUIZ_RETRIEVAL_PRACTICE.value:
            quiz = row.get("quiz_activity")
            if not bool(row.get("quiz_available", False)):
                L[i, 4] = 0
            elif not _is_missing(quiz):
                quiz = float(quiz)
                L[i, 4] = 3 if quiz < 0.20 else (2 if quiz < 0.40 else 1)

        if not eligible:
            L[i, 5] = 0

        L[i, 6] = lookup.get((str(row["case_id"]), action), ABSTAIN)

    validate_vote_matrix(L, SOURCES)
    return L, feasible


def _family_count(matrix: np.ndarray) -> np.ndarray:
    counts = np.zeros(len(matrix), dtype=int)
    for i, votes in enumerate(matrix):
        families = {
            SOURCES[j].family
            for j, vote in enumerate(votes)
            if vote != ABSTAIN
        }
        counts[i] = len(families)
    return counts


def _retention_metadata(family_count: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return the frozen source-support decision without discarding audit rows."""

    counts = np.asarray(family_count, dtype=int)
    if counts.ndim != 1:
        raise ValueError("family_count must be one-dimensional")
    retained = counts >= MINIMUM_INDEPENDENT_SOURCE_FAMILIES
    status = np.where(
        retained,
        RETAINED_LABEL_STATUS,
        INSUFFICIENT_SUPPORT_LABEL_STATUS,
    )
    return retained, status


def run() -> int:
    frozen = _load_frozen()
    candidates, case_lineage_source = _select_panel_a_candidates(frozen)
    votes, feasible = evaluate_votes(candidates, frozen)

    frozen_keys = set(
        zip(frozen["case_id"].astype(str), frozen["action_id"].astype(str))
    )
    feasible_keys = set(
        zip(
            candidates.loc[feasible, "case_id"].astype(str),
            candidates.loc[feasible, "action_id"].astype(str),
        )
    )
    if frozen_keys != feasible_keys:
        missing_reviews = feasible_keys - frozen_keys
        extra_reviews = frozen_keys - feasible_keys
        raise RuntimeError(
            "FEASIBILITY_REVIEW_KEY_MISMATCH "
            f"missing={len(missing_reviews)} extra={len(extra_reviews)}"
        )

    folds = sorted(int(x) for x in pd.unique(candidates["outer_fold"]))
    if len(folds) < 2:
        raise RuntimeError("AT_LEAST_TWO_OUTER_FOLDS_REQUIRED")

    probs = np.full((len(candidates), CARDINALITY), np.nan, dtype=float)
    fold_manifests = []

    for fold in folds:
        train_mask = candidates["outer_fold"].astype(int).to_numpy() != fold
        holdout_mask = ~train_mask
        train_votes = votes[train_mask]
        holdout_votes = votes[holdout_mask]

        model = fit_label_model(
            train_votes,
            SOURCES,
            seed=SEED_BASE + fold,
            epochs=EPOCHS,
        )
        holdout_probs = np.asarray(
            model.predict_proba(L=holdout_votes),
            dtype=float,
        )
        if holdout_probs.shape != (int(holdout_mask.sum()), CARDINALITY):
            raise RuntimeError(f"UNEXPECTED_PROBABILITY_SHAPE_FOLD={fold}")
        probs[holdout_mask] = holdout_probs

        weights = [float(x) for x in model.get_weights().tolist()]
        fold_manifests.append(
            {
                "outer_fold": fold,
                "fit_scope": "PANEL_A_OUTER_FOLD_TRAIN_ONLY",
                "seed": SEED_BASE + fold,
                "epochs": EPOCHS,
                "train_rows": int(train_mask.sum()),
                "holdout_rows": int(holdout_mask.sum()),
                "train_case_count": int(
                    candidates.loc[train_mask, "case_id"].nunique()
                ),
                "holdout_case_count": int(
                    candidates.loc[holdout_mask, "case_id"].nunique()
                ),
                "source_weights": {
                    source.name: weight
                    for source, weight in zip(SOURCES, weights)
                },
            }
        )

    if np.isnan(probs).any():
        raise RuntimeError("OOF_PROBABILITIES_INCOMPLETE")
    if not np.allclose(probs.sum(axis=1), 1.0, atol=1e-6):
        raise RuntimeError("OOF_PROBABILITIES_DO_NOT_SUM_TO_ONE")

    expected = probs @ np.arange(CARDINALITY, dtype=float)
    confidence = probs.max(axis=1)
    entropy = -np.sum(
        probs * np.log(np.clip(probs, 1e-12, 1.0)),
        axis=1,
    )
    family_count = _family_count(votes)
    retained_for_training, label_status = _retention_metadata(family_count)

    labels = candidates[
        ["query_id", "case_id", "outer_fold", "stage", "action_id"]
    ].copy()
    labels["eligible"] = feasible
    for class_id in range(CARDINALITY):
        labels[f"probability_relevance_{class_id}"] = probs[:, class_id]
    labels["expected_relevance"] = expected
    labels["hard_relevance"] = probs.argmax(axis=1).astype(int)
    labels["label_confidence"] = confidence
    labels["label_entropy"] = entropy
    labels["independent_source_families"] = family_count
    labels["retained_for_training"] = retained_for_training
    labels["external_review_present"] = [
        (str(cid), str(action)) in frozen_keys
        for cid, action in zip(labels["case_id"], labels["action_id"])
    ]
    labels["label_status"] = label_status

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    labels_path = OUTPUT_DIR / "probabilistic_relevance_labels.parquet"
    votes_path = OUTPUT_DIR / "weak_vote_matrix.parquet"
    corr_path = OUTPUT_DIR / "source_correlation_audit.csv"
    manifest_path = OUTPUT_DIR / "label_model_manifest.json"

    labels.to_parquet(labels_path, index=False)

    votes_df = labels[
        ["query_id", "case_id", "outer_fold", "stage", "action_id"]
    ].copy()
    for j, source in enumerate(SOURCES):
        votes_df[source.name] = votes[:, j]
    votes_df.to_parquet(votes_path, index=False)

    correlation = source_correlation_audit(votes, SOURCES)
    correlation.to_csv(corr_path, index=False)

    manifest = {
        "schema_version": "panel_a_snorkel_oof_v4",
        "status": "PASS",
        "panel": "A",
        "panel_b_touched": False,
        "runtime_authorized": False,
        "frozen_panel_a_sha256": EXPECTED_FROZEN_SHA256,
        "case_lineage_sha256": EXPECTED_CASE_LINEAGE_SHA256,
        "case_lineage_source": case_lineage_source,
        "cardinality": CARDINALITY,
        "case_count": int(labels["case_id"].nunique()),
        "action_row_count": int(len(labels)),
        "external_review_record_count": int(len(frozen)),
        "eligible_action_row_count": int(feasible.sum()),
        "minimum_independent_source_families": (
            MINIMUM_INDEPENDENT_SOURCE_FAMILIES
        ),
        "retained_row_count": int(retained_for_training.sum()),
        "insufficient_support_row_count": int((~retained_for_training).sum()),
        "eligible_retained_row_count": int(
            (feasible & retained_for_training).sum()
        ),
        "eligible_insufficient_support_row_count": int(
            (feasible & ~retained_for_training).sum()
        ),
        "external_review_key_count": int(len(frozen_keys)),
        "outer_folds": folds,
        "fit_protocol": "OUTER_FOLD_TRAIN_ONLY",
        "sources": [
            {"name": source.name, "family": source.family}
            for source in SOURCES
        ],
        "fold_models": fold_manifests,
        "mean_expected_relevance": float(expected.mean()),
        "mean_confidence": float(confidence.mean()),
        "mean_entropy": float(entropy.mean()),
        "source_family_count_distribution": {
            str(k): int(v)
            for k, v in sorted(Counter(family_count.tolist()).items())
        },
        "labels_sha256": _sha256(labels_path),
        "votes_sha256": _sha256(votes_path),
        "correlation_audit_sha256": _sha256(corr_path),
        "scientific_constraints": [
            "Panel B files are not read by this runner.",
            "Snorkel cardinality is fixed at 4 for relevance classes 0..3.",
            "Each fold model is fit on Panel A rows outside the held-out outer fold.",
            "No fallback or fabricated metric is permitted on fitting failure.",
            "External Gemini reviews are one provenance-preserving source, not three synthetic reviewers.",
            "All OOF probabilities are retained for audit, but rows with fewer than two independent source families are not supervised targets.",
            "RUNTIME_AUTHORIZED remains FALSE.",
        ],
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    print("=== PANEL A SNORKEL OOF FIT ===")
    print(f"FROZEN_PANEL_A_SHA256={EXPECTED_FROZEN_SHA256}")
    print(f"CARDINALITY={CARDINALITY}")
    print(f"PANEL_A_CASES={labels['case_id'].nunique()}")
    print(f"ACTION_ROWS={len(labels)}")
    print(f"ELIGIBLE_ACTION_ROWS={int(feasible.sum())}")
    print(f"MINIMUM_INDEPENDENT_SOURCE_FAMILIES={MINIMUM_INDEPENDENT_SOURCE_FAMILIES}")
    print(f"RETAINED_ROWS={int(retained_for_training.sum())}")
    print(f"INSUFFICIENT_SUPPORT_ROWS={int((~retained_for_training).sum())}")
    print(
        "ELIGIBLE_INSUFFICIENT_SUPPORT_ROWS="
        f"{int((feasible & ~retained_for_training).sum())}"
    )
    print(f"EXTERNAL_REVIEW_RECORDS={len(frozen)}")
    print(f"OUTER_FOLDS={','.join(map(str, folds))}")
    print("FIT_PROTOCOL=OUTER_FOLD_TRAIN_ONLY")
    print(f"MEAN_EXPECTED_RELEVANCE={expected.mean():.6f}")
    print(f"MEAN_CONFIDENCE={confidence.mean():.6f}")
    print(f"MEAN_ENTROPY={entropy.mean():.6f}")
    print("PANEL_B_TOUCHED=FALSE")
    print("RUNTIME_AUTHORIZED=FALSE")
    print("SNORKEL_PANEL_A_OOF=PASS")
    print("NEXT_ACTION=TRAIN_5_EBM_ACTION_MODELS")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.parse_args()
    return run()


if __name__ == "__main__":
    raise SystemExit(main())
