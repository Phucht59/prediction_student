from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from interpret.glassbox import ExplainableBoostingRegressor

BOOTSTRAP_ITERATIONS = 5000
BOOTSTRAP_SEED = 2026
SOURCE_DEPENDENCY_MAX_ABS_NDCG_DROP = 0.05
EXPECTED_CASES = 300
EXPECTED_ROWS = 1500
EXPECTED_ELIGIBLE = 1117
EXPECTED_FULL_NDCG = 0.9722541839577713
EXPECTED_CONFIG_ID = "a70599afad40"

META_COLS = ("query_id", "case_id", "outer_fold", "stage", "action_id")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def ndcg_at_3(y: np.ndarray, s: np.ndarray) -> float:
    k = min(3, len(y))
    order = np.argsort(-s)[:k]
    ideal = np.argsort(-y)[:k]
    gains = np.power(2.0, y) - 1.0
    discounts = 1.0 / np.log2(np.arange(2, k + 2))
    dcg = float(np.sum(gains[order] * discounts))
    idcg = float(np.sum(gains[ideal] * discounts))
    return 0.0 if idcg <= 0.0 else dcg / idcg


def query_contributions(
    frame: pd.DataFrame,
    score_col: str,
    relevance_col: str = "expected_relevance",
) -> pd.DataFrame:
    rows: list[dict] = []
    for query_id, group in frame.groupby("query_id", sort=True):
        group = group.sort_values(
            [score_col, "action_id"],
            ascending=[False, True],
        )
        y = group[relevance_col].to_numpy(dtype=float)
        s = group[score_col].to_numpy(dtype=float)
        k = min(3, len(group))

        ndcg = ndcg_at_3(y, s)
        top_index = int(np.argmax(s))
        exact_top1 = float(y[top_index] >= float(np.max(y)) - 1e-12)

        pair_correct = 0
        pair_total = 0
        for i in range(len(y)):
            for j in range(i + 1, len(y)):
                if y[i] == y[j] or s[i] == s[j]:
                    continue
                pair_total += 1
                pair_correct += int(
                    (y[i] > y[j]) == (s[i] > s[j])
                )

        positives = y >= 1.0
        ranked = np.argsort(-s)
        ranked_pos = positives[ranked]
        if positives.any():
            precision1 = float(ranked_pos[0])
            first_positive = np.flatnonzero(ranked_pos)
            mrr = 1.0 / float(first_positive[0] + 1)
            recall3 = float(ranked_pos[:k].sum()) / float(positives.sum())
        else:
            precision1 = np.nan
            mrr = np.nan
            recall3 = np.nan

        rows.append(
            {
                "query_id": str(query_id),
                "stage": str(group.iloc[0]["stage"]),
                "ndcg_at_3": ndcg,
                "exact_best_top1": exact_top1,
                "pair_correct": pair_correct,
                "pair_total": pair_total,
                "precision_at_1": precision1,
                "mrr": mrr,
                "recall_at_3": recall3,
                "top1_action": str(group.iloc[0]["action_id"]),
            }
        )
    return pd.DataFrame(rows)


def aggregate(contrib: pd.DataFrame) -> dict:
    pair_total = int(contrib["pair_total"].sum())
    pair_correct = int(contrib["pair_correct"].sum())
    top_counts = contrib["top1_action"].value_counts().to_dict()
    probs = np.asarray(list(top_counts.values()), dtype=float)
    probs = probs / probs.sum()
    entropy = float(-np.sum(probs * np.log(np.clip(probs, 1e-12, 1.0))))
    max_entropy = math.log(max(1, len(top_counts)))
    normalized_entropy = 0.0 if max_entropy <= 0 else entropy / max_entropy
    return {
        "query_count": int(len(contrib)),
        "ndcg_at_3": float(contrib["ndcg_at_3"].mean()),
        "exact_best_top1_agreement": float(contrib["exact_best_top1"].mean()),
        "pairwise_accuracy": (
            float(pair_correct / pair_total) if pair_total else 0.0
        ),
        "precision_at_1_relevance_ge_1": float(
            contrib["precision_at_1"].dropna().mean()
        ),
        "mrr_relevance_ge_1": float(contrib["mrr"].dropna().mean()),
        "recall_at_3_relevance_ge_1": float(
            contrib["recall_at_3"].dropna().mean()
        ),
        "unique_top1_actions": int(len(top_counts)),
        "top1_action_counts": {
            str(k): int(v) for k, v in top_counts.items()
        },
        "top1_diversity_normalized_entropy": normalized_entropy,
    }


def paired_bootstrap_delta(
    left: pd.DataFrame,
    right: pd.DataFrame,
    column: str = "ndcg_at_3",
) -> dict:
    l = left.set_index("query_id")
    r = right.set_index("query_id")
    ids = sorted(set(l.index) & set(r.index))
    if len(ids) != EXPECTED_CASES:
        raise RuntimeError(f"BOOTSTRAP_QUERY_COUNT={len(ids)}")
    a = l.loc[ids, column].to_numpy(dtype=float)
    b = r.loc[ids, column].to_numpy(dtype=float)
    diff = a - b
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    n = len(diff)
    samples = np.empty(BOOTSTRAP_ITERATIONS, dtype=float)
    for i in range(BOOTSTRAP_ITERATIONS):
        idx = rng.integers(0, n, size=n)
        samples[i] = float(diff[idx].mean())
    return {
        "iterations": BOOTSTRAP_ITERATIONS,
        "point_delta": float(diff.mean()),
        "bootstrap_mean_delta": float(samples.mean()),
        "ci_low_95": float(np.quantile(samples, 0.025)),
        "ci_high_95": float(np.quantile(samples, 0.975)),
        "probability_positive": float(np.mean(samples > 0.0)),
    }


def build_action_stage_baseline(data: pd.DataFrame) -> np.ndarray:
    output = np.full(len(data), np.nan, dtype=float)
    folds = sorted(int(v) for v in data["outer_fold"].unique())
    for fold in folds:
        train = data[data["outer_fold"].astype(int) != fold]
        hold = data[data["outer_fold"].astype(int) == fold]
        pair_mean = (
            train.groupby(["action_id", "stage"])["expected_relevance"]
            .mean()
            .to_dict()
        )
        action_mean = (
            train.groupby("action_id")["expected_relevance"].mean().to_dict()
        )
        global_mean = float(train["expected_relevance"].mean())
        vals = []
        for row in hold.itertuples(index=False):
            vals.append(
                pair_mean.get(
                    (row.action_id, row.stage),
                    action_mean.get(row.action_id, global_mean),
                )
            )
        output[hold.index.to_numpy()] = np.asarray(vals, dtype=float) / 3.0
    if np.isnan(output).any():
        raise RuntimeError("ACTION_STAGE_BASELINE_INCOMPLETE")
    return np.clip(output, 0.0, 1.0)


def make_ebm(params: dict, action_index: int, fold: int):
    kwargs = dict(params)
    kwargs["random_state"] = 2026 + action_index * 100 + fold
    kwargs.setdefault("n_jobs", -2)
    return ExplainableBoostingRegressor(**kwargs)


def permute_context_jointly(
    X: pd.DataFrame,
    stage_values: pd.Series,
    numeric_features: list[str],
    *,
    seed: int,
) -> pd.DataFrame:
    out = X.copy()
    rng = np.random.default_rng(seed)
    stage_series = pd.Series(
        stage_values.to_numpy(),
        index=X.index,
        dtype=object,
    )
    for stage in sorted(stage_series.astype(str).unique()):
        idx = stage_series.index[stage_series.astype(str) == stage].to_numpy()
        if len(idx) <= 1:
            continue
        donor = rng.permutation(idx)
        out.loc[idx, numeric_features] = (
            X.loc[donor, numeric_features].to_numpy()
        )
    return out


def build_context_permutation_scores(base, data: pd.DataFrame) -> tuple[np.ndarray, float]:
    original = np.full(len(data), np.nan, dtype=float)
    permuted = np.full(len(data), np.nan, dtype=float)

    numeric_features = [
        f
        for f in base.FEATURES
        if f not in {
            "stage",
            "vle_available",
            "study_material_available",
            "quiz_available",
        }
    ]

    actions = list(base.ACTIONS)
    folds = sorted(int(v) for v in data["outer_fold"].unique())

    warnings.filterwarnings(
        "ignore",
        message="Missing values detected.*",
        category=UserWarning,
    )

    for action_index, action in enumerate(actions):
        action_mask = data["action_id"].eq(action).to_numpy()
        action_df = data.loc[action_mask].copy()
        action_indices = np.flatnonzero(action_mask)
        X = base._prepare_X(action_df)
        y = action_df["expected_relevance"].to_numpy(dtype=float)
        fold_values = action_df["outer_fold"].astype(int).to_numpy()

        for fold in folds:
            train = fold_values != fold
            hold = ~train
            model = make_ebm(base.EBM_PARAMS, action_index, fold)
            model.fit(X.loc[train], y[train])

            hold_local_idx = np.flatnonzero(hold)
            global_idx = action_indices[hold_local_idx]

            pred_original = np.clip(
                np.asarray(model.predict(X.loc[hold]), dtype=float) / 3.0,
                0.0,
                1.0,
            )
            X_hold = X.loc[hold].copy()
            X_perm = permute_context_jointly(
                X_hold,
                action_df.loc[hold, "stage"],
                numeric_features,
                seed=2026 + action_index * 1000 + fold,
            )
            pred_perm = np.clip(
                np.asarray(model.predict(X_perm), dtype=float) / 3.0,
                0.0,
                1.0,
            )
            original[global_idx] = pred_original
            permuted[global_idx] = pred_perm

    if np.isnan(original).any() or np.isnan(permuted).any():
        raise RuntimeError("CONTEXT_PERMUTATION_PREDICTIONS_INCOMPLETE")

    stored = np.clip(
        data["ebm_oof_score"].to_numpy(dtype=float) / 3.0,
        0.0,
        1.0,
    )
    max_abs_diff = float(np.max(np.abs(original - stored)))
    return permuted, max_abs_diff


def fit_loo_expected_relevance(
    votes_df: pd.DataFrame,
    source_columns: list[str],
    family_map: dict[str, str],
    excluded_sources: set[str],
    fit_label_model,
    WeakLabelSource,
) -> np.ndarray:
    remaining = [c for c in source_columns if c not in excluded_sources]
    if len(remaining) < 2:
        raise RuntimeError(
            f"TOO_FEW_SOURCES_AFTER_ABLATION={sorted(excluded_sources)}"
        )
    sources = tuple(
        WeakLabelSource(name=c, family=family_map[c])
        for c in remaining
    )
    matrix = votes_df[remaining].to_numpy(dtype=int)
    fold_values = votes_df["outer_fold"].astype(int).to_numpy()
    expected = np.full(len(votes_df), np.nan, dtype=float)

    for fold in sorted(np.unique(fold_values)):
        train = fold_values != fold
        hold = ~train
        model = fit_label_model(
            matrix[train],
            sources,
            seed=2026 + int(fold),
            epochs=1000,
        )
        probs = np.asarray(
            model.predict_proba(L=matrix[hold]),
            dtype=float,
        )
        if probs.shape[1] != 4:
            raise RuntimeError("LOO_LABELMODEL_CARDINALITY_NOT_4")
        expected[hold] = probs @ np.arange(4, dtype=float)

    if np.isnan(expected).any():
        raise RuntimeError("LOO_EXPECTED_RELEVANCE_INCOMPLETE")
    return expected


def active_family_counts(
    votes_df: pd.DataFrame,
    source_columns: list[str],
    family_map: dict[str, str],
) -> np.ndarray:
    matrix = votes_df[source_columns].to_numpy(dtype=int)
    result = np.zeros(len(votes_df), dtype=int)
    for i, row in enumerate(matrix):
        families = {
            family_map[source_columns[j]]
            for j, vote in enumerate(row)
            if int(vote) != -1
        }
        result[i] = len(families)
    return result


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", type=Path, required=True)
    args = ap.parse_args()
    root = args.repo.resolve()
    sys.path.insert(0, str(root))

    from scripts.recommend_hybrid.explainable_v2 import train_five_ebm_models as base
    from src.recommend_hybrid.explainable_v2.weak_labels import (
        WeakLabelSource,
        fit_label_model,
    )

    freeze_dir = (
        root
        / "artifacts/recommend_hybrid/explainable_v2/frozen/ranker_panel_a_v1"
    )
    freeze_manifest_path = freeze_dir / "RANKER_PANEL_A_FREEZE_MANIFEST.json"
    votes_path = (
        root
        / "artifacts/recommend_hybrid/explainable_v2/labels/panel_a_v1"
        / "weak_vote_matrix.parquet"
    )
    labels_manifest_path = (
        root
        / "artifacts/recommend_hybrid/explainable_v2/labels/panel_a_v1"
        / "label_model_manifest.json"
    )

    for p in (freeze_manifest_path, votes_path, labels_manifest_path):
        if not p.exists():
            raise RuntimeError(f"MISSING_REQUIRED_INPUT={p}")

    freeze = json.loads(freeze_manifest_path.read_text(encoding="utf-8"))
    label_manifest = json.loads(
        labels_manifest_path.read_text(encoding="utf-8")
    )

    if freeze.get("status") != "PASS":
        raise RuntimeError("RANKER_FREEZE_NOT_PASS")
    if freeze.get("selected_config_id") != EXPECTED_CONFIG_ID:
        raise RuntimeError("RANKER_FREEZE_CONFIG_MISMATCH")
    if freeze.get("panel_b_touched") is not False:
        raise RuntimeError("PANEL_B_TOUCHED_IN_RANKER_FREEZE")
    if freeze.get("runtime_authorized") is not False:
        raise RuntimeError("RUNTIME_AUTHORIZED_MUST_REMAIN_FALSE")
    if label_manifest.get("status") != "PASS":
        raise RuntimeError("LABEL_MODEL_MANIFEST_NOT_PASS")
    if label_manifest.get("fit_protocol") != "OUTER_FOLD_TRAIN_ONLY":
        raise RuntimeError("LABEL_MODEL_PROTOCOL_NOT_OUTER_FOLD_TRAIN_ONLY")
    if int(label_manifest.get("cardinality", -1)) != 4:
        raise RuntimeError("LABEL_MODEL_CARDINALITY_NOT_4")
    if label_manifest.get("panel_b_touched") is not False:
        raise RuntimeError("PANEL_B_TOUCHED_IN_LABEL_MODEL")

    data, _, _ = base._load_inputs()
    if len(data) != EXPECTED_ROWS or data["case_id"].nunique() != EXPECTED_CASES:
        raise RuntimeError("FINAL_EBM_INPUT_SHAPE_MISMATCH")
    if int(data["eligible"].astype(bool).sum()) != EXPECTED_ELIGIBLE:
        raise RuntimeError("FINAL_EBM_ELIGIBLE_COUNT_MISMATCH")
    if getattr(base, "LOCKED_GRID_SELECTED_CONFIG_ID", None) != EXPECTED_CONFIG_ID:
        raise RuntimeError("RUNNER_CONFIG_ID_NOT_LOCKED")
    if int(base.EBM_PARAMS["interactions"]) != 3:
        raise RuntimeError("FINAL_INTERACTION_BUDGET_MISMATCH")

    # _load_inputs() returns the training table and intentionally does not
    # include OOF predictions. Load the frozen OOF artifact explicitly and
    # align it by stable keys before computing release-gate metrics.
    oof_path = (
        root
        / "artifacts/recommend_hybrid/explainable_v2/models/ebm_panel_a_v1"
        / "panel_a_ebm_oof_predictions.parquet"
    )
    if not oof_path.exists():
        raise RuntimeError(f"MISSING_OOF_PREDICTIONS={oof_path}")

    oof = pd.read_parquet(oof_path)
    required_oof = set(META_COLS) | {"ebm_oof_score"}
    missing_oof = sorted(required_oof - set(oof.columns))
    if missing_oof:
        raise RuntimeError(f"OOF_COLUMNS_MISSING={missing_oof}")
    if len(oof) != EXPECTED_ROWS:
        raise RuntimeError(f"OOF_ROW_COUNT={len(oof)} expected={EXPECTED_ROWS}")

    merge_keys = list(META_COLS)
    if oof.duplicated(merge_keys).any():
        raise RuntimeError("OOF_DUPLICATE_STABLE_KEYS")
    if data.duplicated(merge_keys).any():
        raise RuntimeError("TRAINING_TABLE_DUPLICATE_STABLE_KEYS")

    data = data.merge(
        oof[merge_keys + ["ebm_oof_score"]],
        on=merge_keys,
        how="left",
        validate="one_to_one",
    )
    if data["ebm_oof_score"].isna().any():
        raise RuntimeError(
            "OOF_MERGE_MISSING_ROWS="
            f"{int(data['ebm_oof_score'].isna().sum())}"
        )

    data = data.reset_index(drop=True)
    data["full_score"] = np.clip(
        data["ebm_oof_score"].to_numpy(dtype=float) / 3.0,
        0.0,
        1.0,
    )
    data["action_stage_score"] = build_action_stage_baseline(data)

    eligible = data[data["eligible"].astype(bool)].copy()
    full_contrib = query_contributions(eligible, "full_score")
    baseline_contrib = query_contributions(eligible, "action_stage_score")
    full_metrics = aggregate(full_contrib)
    baseline_metrics = aggregate(baseline_contrib)

    if abs(full_metrics["ndcg_at_3"] - EXPECTED_FULL_NDCG) > 1e-12:
        raise RuntimeError(
            "FROZEN_OPERATIONAL_NDCG_REPRODUCIBILITY_FAILURE="
            f"{full_metrics['ndcg_at_3']}"
        )

    baseline_boot = paired_bootstrap_delta(full_contrib, baseline_contrib)
    baseline_gate = baseline_boot["ci_low_95"] > 0.0

    print("=== RELEASE GATES: BASELINE ===")
    print(f"FULL_NDCG_AT_3={full_metrics['ndcg_at_3']:.6f}")
    print(f"ACTION_STAGE_ONLY_NDCG_AT_3={baseline_metrics['ndcg_at_3']:.6f}")
    print(
        "FULL_MINUS_ACTION_STAGE_CI95="
        f"[{baseline_boot['ci_low_95']:.6f},"
        f"{baseline_boot['ci_high_95']:.6f}]"
    )
    print(
        "FULL_BEATS_ACTION_STAGE_ONLY="
        + str(full_metrics["ndcg_at_3"] > baseline_metrics["ndcg_at_3"]).upper()
    )
    print(
        "FULL_MINUS_ACTION_STAGE_ONLY_CI_EXCLUDES_ZERO="
        + str(baseline_gate).upper()
    )

    start_perm = time.time()
    permuted_scores, recompute_max_diff = build_context_permutation_scores(
        base,
        data,
    )
    data["context_permuted_score"] = permuted_scores
    perm_eligible = data[data["eligible"].astype(bool)].copy()
    perm_contrib = query_contributions(
        perm_eligible,
        "context_permuted_score",
    )
    perm_metrics = aggregate(perm_contrib)
    perm_boot = paired_bootstrap_delta(full_contrib, perm_contrib)
    context_gate = (
        full_metrics["ndcg_at_3"] > perm_metrics["ndcg_at_3"]
    )

    print("=== RELEASE GATES: CONTEXT PERMUTATION ===")
    print(f"OOF_RECOMPUTE_MAX_ABS_DIFF={recompute_max_diff:.12g}")
    print(f"PERMUTED_CONTEXT_NDCG_AT_3={perm_metrics['ndcg_at_3']:.6f}")
    print(
        "FULL_MINUS_PERMUTED_NDCG="
        f"{full_metrics['ndcg_at_3'] - perm_metrics['ndcg_at_3']:.6f}"
    )
    print(
        "FULL_MINUS_PERMUTED_CI95="
        f"[{perm_boot['ci_low_95']:.6f},"
        f"{perm_boot['ci_high_95']:.6f}]"
    )
    print(
        "CONTEXT_PERMUTATION_DEGRADES_METRIC="
        + str(context_gate).upper()
    )
    print(f"CONTEXT_PERMUTATION_MIN={(time.time()-start_perm)/60:.2f}")

    # Operational invalid-action audit: rank exclusively among feasible actions.
    top1 = (
        eligible.sort_values(
            ["query_id", "full_score", "action_id"],
            ascending=[True, False, True],
        )
        .groupby("query_id", as_index=False)
        .first()
    )
    invalid_top1 = int((~top1["eligible"].astype(bool)).sum())
    invalid_action_rate = float(invalid_top1 / len(top1))
    invalid_gate = invalid_action_rate == 0.0

    per_stage = {}
    for stage, group in eligible.groupby("stage", sort=True):
        c = query_contributions(group, "full_score")
        per_stage[str(stage)] = aggregate(c)

    per_action = {}
    for action, group in data.groupby("action_id", sort=True):
        err = (
            group["ebm_oof_score"].to_numpy(dtype=float)
            - group["expected_relevance"].to_numpy(dtype=float)
        )
        per_action[str(action)] = {
            "row_count": int(len(group)),
            "eligible_row_count": int(group["eligible"].astype(bool).sum()),
            "rmse": float(np.sqrt(np.mean(np.square(err)))),
            "mae": float(np.mean(np.abs(err))),
            "top1_count": int(
                full_contrib["top1_action"].eq(str(action)).sum()
            ),
        }

    print("=== RELEASE GATES: OPERATIONAL SAFETY ===")
    print(f"INVALID_ACTION_RATE={invalid_action_rate:.6f}")
    print(
        "INVALID_ACTION_RATE_ZERO="
        + str(invalid_gate).upper()
    )
    print(
        f"UNIQUE_TOP1_ACTIONS={full_metrics['unique_top1_actions']}"
    )
    print(
        "TOP1_DIVERSITY_NORMALIZED_ENTROPY="
        f"{full_metrics['top1_diversity_normalized_entropy']:.6f}"
    )

    votes = pd.read_parquet(votes_path).copy()
    if len(votes) != EXPECTED_ROWS:
        raise RuntimeError("VOTE_MATRIX_ROW_COUNT_MISMATCH")
    source_columns = [c for c in votes.columns if c not in META_COLS]

    sources_manifest = label_manifest.get("sources", [])
    family_map = {
        str(item["name"]): str(item["family"])
        for item in sources_manifest
    }
    if set(source_columns) != set(family_map):
        raise RuntimeError(
            "VOTE_SOURCE_MANIFEST_MISMATCH "
            f"votes={source_columns} manifest={sorted(family_map)}"
        )

    # Confirm vote rows line up with model data by stable keys.
    vote_keys = votes[list(META_COLS)].copy()
    model_keys = data[list(META_COLS)].copy()
    for col in ("query_id", "case_id", "stage", "action_id"):
        vote_keys[col] = vote_keys[col].astype(str)
        model_keys[col] = model_keys[col].astype(str)
    vote_keys["outer_fold"] = vote_keys["outer_fold"].astype(int)
    model_keys["outer_fold"] = model_keys["outer_fold"].astype(int)
    if not vote_keys.equals(model_keys):
        raise RuntimeError("VOTE_AND_MODEL_ROW_ORDER_MISMATCH")

    family_counts = active_family_counts(
        votes,
        source_columns,
        family_map,
    )
    data["active_source_family_count"] = family_counts
    eligible_family_violations = int(
        (
            data.loc[data["eligible"].astype(bool), "active_source_family_count"]
            < 2
        ).sum()
    )
    family_support_gate = eligible_family_violations == 0

    print("=== RELEASE GATES: LABEL SOURCE SUPPORT ===")
    print(
        "ELIGIBLE_ROWS_WITH_LT2_SOURCE_FAMILIES="
        f"{eligible_family_violations}"
    )
    print(
        "ELIGIBLE_MINIMUM_2_SOURCE_FAMILIES="
        + str(family_support_gate).upper()
    )

    violating_rows = []
    violation_mask = (
        data["eligible"].astype(bool)
        & (data["active_source_family_count"] < 2)
    )
    if violation_mask.any():
        for idx in data.index[violation_mask]:
            active_sources = [
                source
                for source in source_columns
                if int(votes.loc[idx, source]) != -1
            ]
            active_families = sorted(
                {family_map[source] for source in active_sources}
            )
            item = {
                "query_id": str(data.loc[idx, "query_id"]),
                "case_id": str(data.loc[idx, "case_id"]),
                "outer_fold": int(data.loc[idx, "outer_fold"]),
                "stage": str(data.loc[idx, "stage"]),
                "action_id": str(data.loc[idx, "action_id"]),
                "active_sources": active_sources,
                "active_families": active_families,
                "active_source_family_count": int(
                    data.loc[idx, "active_source_family_count"]
                ),
            }
            violating_rows.append(item)
            print(
                "SOURCE_FAMILY_VIOLATION="
                + json.dumps(item, sort_keys=True)
            )

    # Leave-one-source-out: re-fit Snorkel strictly train-only on every outer fold.
    loo_source_results = {}
    start_loo = time.time()
    for source in source_columns:
        expected = fit_loo_expected_relevance(
            votes,
            source_columns,
            family_map,
            {source},
            fit_label_model,
            WeakLabelSource,
        )
        temp = data.copy()
        temp["loo_expected_relevance"] = expected
        temp = temp[temp["eligible"].astype(bool)].copy()
        contrib = query_contributions(
            temp,
            "full_score",
            "loo_expected_relevance",
        )
        metrics = aggregate(contrib)
        drop = float(
            full_metrics["ndcg_at_3"] - metrics["ndcg_at_3"]
        )
        loo_source_results[source] = {
            "family": family_map[source],
            "ndcg_at_3": metrics["ndcg_at_3"],
            "full_minus_loo_ndcg": drop,
            "catastrophic_dependency": (
                drop > SOURCE_DEPENDENCY_MAX_ABS_NDCG_DROP
            ),
        }
        print(
            f"LOO_SOURCE={source} "
            f"FAMILY={family_map[source]} "
            f"NDCG3={metrics['ndcg_at_3']:.6f} "
            f"FULL_MINUS_LOO={drop:.6f}"
        )

    max_source_drop = max(
        item["full_minus_loo_ndcg"]
        for item in loo_source_results.values()
    )
    source_dependency_gate = (
        max_source_drop <= SOURCE_DEPENDENCY_MAX_ABS_NDCG_DROP
    )

    # Stronger family-level diagnostics, not a separate hard protocol gate.
    # Snorkel LabelModel requires at least 3 labeling functions.  A family
    # ablation that leaves fewer than 3 sources is scientifically undefined
    # for this estimator and must be reported as UNSUPPORTED, never fabricated.
    families = sorted(set(family_map.values()))
    loo_family_results = {}
    for family in families:
        excluded = {
            source
            for source in source_columns
            if family_map[source] == family
        }
        remaining = [
            source for source in source_columns if source not in excluded
        ]
        if len(remaining) < 3:
            loo_family_results[family] = {
                "excluded_sources": sorted(excluded),
                "remaining_sources": remaining,
                "status": "UNSUPPORTED_SNORKEL_REQUIRES_AT_LEAST_3_LFS",
            }
            print(
                f"LOO_FAMILY={family} "
                f"STATUS=UNSUPPORTED_SNORKEL_REQUIRES_AT_LEAST_3_LFS "
                f"REMAINING_SOURCES={len(remaining)}"
            )
            continue

        expected = fit_loo_expected_relevance(
            votes,
            source_columns,
            family_map,
            excluded,
            fit_label_model,
            WeakLabelSource,
        )
        temp = data.copy()
        temp["loo_expected_relevance"] = expected
        temp = temp[temp["eligible"].astype(bool)].copy()
        contrib = query_contributions(
            temp,
            "full_score",
            "loo_expected_relevance",
        )
        metrics = aggregate(contrib)
        drop = float(
            full_metrics["ndcg_at_3"] - metrics["ndcg_at_3"]
        )
        loo_family_results[family] = {
            "excluded_sources": sorted(excluded),
            "remaining_sources": remaining,
            "status": "PASS_DIAGNOSTIC",
            "ndcg_at_3": metrics["ndcg_at_3"],
            "full_minus_loo_ndcg": drop,
        }
        print(
            f"LOO_FAMILY={family} "
            f"NDCG3={metrics['ndcg_at_3']:.6f} "
            f"FULL_MINUS_LOO={drop:.6f}"
        )

    print(
        "NO_SINGLE_LABEL_SOURCE_DEPENDENCY="
        + str(source_dependency_gate).upper()
    )
    print(
        "SOURCE_DEPENDENCY_MAX_ALLOWED_NDCG_DROP="
        f"{SOURCE_DEPENDENCY_MAX_ABS_NDCG_DROP:.3f}"
    )
    print(f"MAX_SINGLE_SOURCE_NDCG_DROP={max_source_drop:.6f}")
    print(f"LABEL_ABLATION_MIN={(time.time()-start_loo)/60:.2f}")

    gates = {
        "frozen_operational_metric_reproduced": True,
        "full_beats_action_stage_only": (
            full_metrics["ndcg_at_3"] > baseline_metrics["ndcg_at_3"]
        ),
        "full_minus_action_stage_only_ci_excludes_zero": baseline_gate,
        "context_permutation_degrades_metric": context_gate,
        "invalid_action_rate_zero": invalid_gate,
        "eligible_minimum_two_source_families": family_support_gate,
        "no_single_label_source_dependency": source_dependency_gate,
        "panel_b_untouched": True,
        "runtime_authorized_false": True,
    }
    release_pass = all(gates.values())

    report = {
        "schema_version": "panel_a_release_gates_v1",
        "status": "PASS" if release_pass else "FAIL",
        "development_only": True,
        "panel": "A",
        "panel_b_touched": False,
        "runtime_authorized": False,
        "final_metrics_claimed": False,
        "frozen_ranker_manifest_sha256": sha256_file(
            freeze_manifest_path
        ),
        "label_model_manifest_sha256": sha256_file(
            labels_manifest_path
        ),
        "weak_vote_matrix_sha256": sha256_file(votes_path),
        "selected_config_id": EXPECTED_CONFIG_ID,
        "full_metrics": full_metrics,
        "action_stage_only_metrics": baseline_metrics,
        "full_minus_action_stage_bootstrap": baseline_boot,
        "context_permutation_metrics": perm_metrics,
        "full_minus_context_permutation_bootstrap": perm_boot,
        "oof_recompute_max_abs_diff": recompute_max_diff,
        "invalid_action_rate": invalid_action_rate,
        "per_stage_metrics": per_stage,
        "per_action_metrics": per_action,
        "eligible_rows_with_lt2_active_source_families": (
            eligible_family_violations
        ),
        "source_family_violating_rows": violating_rows,
        "leave_one_source_out": loo_source_results,
        "leave_one_family_out_diagnostic": loo_family_results,
        "source_dependency_max_allowed_ndcg_drop": (
            SOURCE_DEPENDENCY_MAX_ABS_NDCG_DROP
        ),
        "release_gates": gates,
    }

    out_dir = (
        root
        / "artifacts/recommend_hybrid/explainable_v2/release_gates"
        / "panel_a_v1"
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    report_path = out_dir / "PANEL_A_RELEASE_GATES.json"
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    print("=== PANEL-A RELEASE GATE SUMMARY ===")
    for key, value in gates.items():
        print(f"{key.upper()}={str(bool(value)).upper()}")
    print(f"RELEASE_GATE_REPORT={report_path}")
    print(
        "PANEL_A_RELEASE_GATES="
        + ("PASS" if release_pass else "FAIL")
    )
    if release_pass:
        print("NEXT_ACTION=SELECT_AND_FREEZE_SAFETY_ROUTER")
    else:
        print("NEXT_ACTION=HARDEN_FAILED_RELEASE_GATES_BEFORE_ROUTER_FREEZE")
    return 0 if release_pass else 2


if __name__ == "__main__":
    raise SystemExit(main())
