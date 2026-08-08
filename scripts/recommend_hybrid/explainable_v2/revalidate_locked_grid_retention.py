"""Revalidate the locked EBM grid after excluding one unsupported query."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[3]
GRID_DIR = (
    ROOT
    / "artifacts/recommend_hybrid/explainable_v2/ranker_development/ebm_locked_grid_v1"
)
LABEL_PATH = (
    ROOT
    / "artifacts/recommend_hybrid/explainable_v2/labels/panel_a_v1"
    / "probabilistic_relevance_labels.parquet"
)
OOF_PATH = (
    ROOT
    / "artifacts/recommend_hybrid/explainable_v2/models/ebm_panel_a_v1"
    / "panel_a_ebm_oof_predictions.parquet"
)
STATE_PATH = GRID_DIR / "search_state.jsonl"
OUTPUT_PATH = GRID_DIR / "EBM_GRID_RETENTION_REVALIDATION.json"
ORIGINAL_SELECTED_CONFIG_ID = "a70599afad40"
BOOTSTRAP_SEED = 2026
BOOTSTRAP_ITERATIONS = 2000


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _ndcg_at_3(group: pd.DataFrame) -> float:
    y = group["expected_relevance"].to_numpy(dtype=float)
    score = group["ebm_oof_score"].to_numpy(dtype=float)
    action = group["action_id"].astype(str).to_numpy()
    k = min(3, len(group))
    discounts = 1.0 / np.log2(np.arange(2, k + 2))
    gains = np.power(2.0, y) - 1.0
    order = np.lexsort((action, -score))[:k]
    ideal = np.lexsort((action, -y))[:k]
    return float(
        np.sum(gains[order] * discounts)
        / np.sum(gains[ideal] * discounts)
    )


def _bootstrap(candidate: np.ndarray, best: np.ndarray) -> dict[str, float]:
    difference = candidate - best
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    samples = np.asarray(
        [
            difference[
                rng.integers(0, len(difference), size=len(difference))
            ].mean()
            for _ in range(BOOTSTRAP_ITERATIONS)
        ],
        dtype=float,
    )
    return {
        "ci_low_95": float(np.quantile(samples, 0.025)),
        "ci_high_95": float(np.quantile(samples, 0.975)),
        "mean_difference_candidate_minus_best": float(samples.mean()),
        "probability_positive": float((samples > 0.0).mean()),
    }


def run() -> int:
    labels = pd.read_parquet(LABEL_PATH)
    unsupported = labels.loc[~labels["retained_for_training"].astype(bool)]
    unsupported_queries = unsupported["query_id"].astype(str).unique().tolist()
    if len(unsupported) != 1 or len(unsupported_queries) != 1:
        raise RuntimeError("EXPECTED_EXACTLY_ONE_UNSUPPORTED_ROW_AND_QUERY")
    excluded_query = unsupported_queries[0]

    oof = pd.read_parquet(OOF_PATH)
    selected_query = labels.merge(
        oof[["query_id", "action_id", "ebm_oof_score"]],
        on=["query_id", "action_id"],
        validate="one_to_one",
    )
    selected_query = selected_query[
        selected_query["query_id"].astype(str).eq(excluded_query)
        & selected_query["eligible"].astype(bool)
    ]
    excluded_selected_ndcg = _ndcg_at_3(selected_query)

    rows = [
        json.loads(line)
        for line in STATE_PATH.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if len(rows) != 432:
        raise RuntimeError(f"LOCKED_GRID_STATE_COUNT={len(rows)} expected=432")
    original_selected = next(
        row for row in rows if row["config_id"] == ORIGINAL_SELECTED_CONFIG_ID
    )
    matching_positions = [
        index
        for index, value in enumerate(original_selected["query_ndcg_at_3"])
        if abs(float(value) - excluded_selected_ndcg) <= 1e-12
    ]
    if len(matching_positions) != 1:
        raise RuntimeError(
            "EXCLUDED_QUERY_POSITION_NOT_UNIQUELY_TRACEABLE="
            f"{matching_positions}"
        )
    excluded_index = matching_positions[0]

    corrected: dict[str, np.ndarray] = {}
    for row in rows:
        contributions = np.asarray(row["query_ndcg_at_3"], dtype=float)
        if len(contributions) != 300:
            raise RuntimeError("LOCKED_GRID_QUERY_CONTRIBUTIONS_INCOMPLETE")
        if abs(float(contributions.mean()) - float(row["ndcg_at_3"])) > 1e-12:
            raise RuntimeError(
                f"LOCKED_GRID_CONTRIBUTION_MEAN_MISMATCH={row['config_id']}"
            )
        corrected[row["config_id"]] = np.delete(contributions, excluded_index)

    empirical_best = max(
        rows,
        key=lambda row: (
            float(corrected[row["config_id"]].mean()),
            row["config_id"],
        ),
    )
    best_contributions = corrected[empirical_best["config_id"]]
    comparisons = []
    indistinguishable = []
    for row in rows:
        bootstrap = _bootstrap(
            corrected[row["config_id"]],
            best_contributions,
        )
        item = {
            "config_id": row["config_id"],
            "complexity_key": row["complexity_key"],
            "corrected_ndcg_at_3": float(
                corrected[row["config_id"]].mean()
            ),
            "bootstrap_vs_empirical_best": bootstrap,
        }
        comparisons.append(item)
        if bootstrap["ci_high_95"] >= 0.0:
            indistinguishable.append(item)

    minimum_complexity = min(
        tuple(item["complexity_key"]) for item in indistinguishable
    )
    simplest = [
        item
        for item in indistinguishable
        if tuple(item["complexity_key"]) == minimum_complexity
    ]
    selected = max(
        simplest,
        key=lambda item: (item["corrected_ndcg_at_3"], item["config_id"]),
    )
    selected_row = next(
        row for row in rows if row["config_id"] == selected["config_id"]
    )

    report = {
        "schema_version": "locked_ebm_grid_retention_revalidation_v1",
        "status": (
            "PASS"
            if selected["config_id"] == ORIGINAL_SELECTED_CONFIG_ID
            else "FAIL_SELECTED_CONFIG_CHANGED"
        ),
        "scope": "PANEL_A_DEVELOPMENT",
        "panel_b_touched": False,
        "runtime_authorized": False,
        "refit_performed": False,
        "exact_reselection_from_stored_per_query_oof": True,
        "selection_rule": (
            "simplest_statistically_indistinguishable_from_empirical_best; "
            "highest_NDCG_within_minimum_complexity_tie"
        ),
        "bootstrap_seed": BOOTSTRAP_SEED,
        "bootstrap_iterations": BOOTSTRAP_ITERATIONS,
        "excluded_query_id": excluded_query,
        "excluded_query_contribution_index": excluded_index,
        "excluded_query_selected_config_ndcg": excluded_selected_ndcg,
        "corrected_query_count": 299,
        "grid_config_count": len(rows),
        "statistically_indistinguishable_config_count": len(indistinguishable),
        "empirical_best": {
            "config_id": empirical_best["config_id"],
            "corrected_ndcg_at_3": float(
                corrected[empirical_best["config_id"]].mean()
            ),
        },
        "selected": {
            key: selected_row[key]
            for key in (
                "config_id",
                "interactions",
                "learning_rate",
                "max_bins",
                "max_rounds",
                "min_samples_leaf",
                "complexity_key",
            )
        }
        | {"corrected_ndcg_at_3": selected["corrected_ndcg_at_3"]},
        "original_selected_config_id": ORIGINAL_SELECTED_CONFIG_ID,
        "selected_config_unchanged": (
            selected["config_id"] == ORIGINAL_SELECTED_CONFIG_ID
        ),
        "lineage": {
            "search_state_sha256": _sha256(STATE_PATH),
            "corrected_labels_sha256": _sha256(LABEL_PATH),
            "historical_selected_oof_sha256": _sha256(OOF_PATH),
        },
        "comparisons": comparisons,
    }
    OUTPUT_PATH.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"EXCLUDED_QUERY={excluded_query}")
    print(f"CORRECTED_QUERY_COUNT={report['corrected_query_count']}")
    print(f"EMPIRICAL_BEST_CONFIG={empirical_best['config_id']}")
    print(f"SELECTED_CONFIG={selected['config_id']}")
    print(
        "SELECTED_CONFIG_UNCHANGED="
        + str(report["selected_config_unchanged"]).upper()
    )
    print("PANEL_B_TOUCHED=FALSE")
    print("RUNTIME_AUTHORIZED=FALSE")
    return 0 if report["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(run())
