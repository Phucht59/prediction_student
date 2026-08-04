"""Retrained negative controls for outcome-grounded V2.1.

Unlike the historical null-score sanity check, every mandatory control here
modifies the registered training signal or feature relationship and refits the
selected outer-fold model. Results are written in resumable batches.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from scientific_core import (
    ACTION_SPECIFIC_FEATURES,
    FeaturePreprocessor,
    RelevanceTransformer,
    STATE_FEATURES,
    aggregate_metrics,
    fit_ranker,
    predict_ranker,
)

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "artifacts/recommend_hybrid/outcome_grounded_v2_1"
DATA = OUT / "dataset"
FINAL = OUT / "final_oof"
CONTROL_OUT = OUT / "negative_controls_retrained"
SEED = 20260804
CONTROLS = [
    "NC1_LABEL_SHUFFLE_RETRAIN",
    "NC2A_TRAIN_STATE_SHUFFLE",
    "NC2B_TEST_STATE_SHUFFLE",
    "NC3_ACTION_IDENTITY_SHUFFLE_RETRAIN",
    "NC4_WRONG_TRAJECTORY_REBUILD",
    "NC5_TIME_REVERSAL_PLACEBO",
]


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def ordered(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.sort_values(["group_id", "action_family"], kind="stable").reset_index(drop=True)


def selected_model(outer_fold: int) -> tuple[str, dict[str, Any]]:
    path = OUT / "model_selection" / f"fold_{outer_fold}_selected.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    return str(payload["model"]), dict(payload["parameters"])


def shuffle_labels(frame: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    result = frame.copy()
    strata = ["stage", "course", "action_family"]
    for _, indices in result.groupby(strata, sort=False).groups.items():
        idx = np.asarray(list(indices))
        permutation = rng.permutation(len(idx))
        result.loc[idx, "continuous_relevance"] = result.loc[
            idx[permutation], "continuous_relevance"
        ].to_numpy()
        result.loc[idx, "graded_relevance"] = result.loc[
            idx[permutation], "graded_relevance"
        ].to_numpy()
    return result


def shuffle_group_state(frame: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    result = frame.copy()
    compatibility = ["stage", "course", "presentation"]
    representatives = result.groupby("group_id", sort=False).first().reset_index()
    indexed = representatives.set_index(representatives["group_id"].astype(str))
    for _, compatible in representatives.groupby(compatibility, sort=False):
        if len(compatible) < 2:
            continue
        source_groups = compatible["group_id"].astype(str).to_numpy()
        donor_groups = source_groups[rng.permutation(len(source_groups))]
        donor_state = indexed[STATE_FEATURES].reindex(donor_groups).reset_index(drop=True)
        mapping = {
            source: donor_state.iloc[position].to_dict()
            for position, source in enumerate(source_groups)
        }
        for source, values in mapping.items():
            mask = result["group_id"].astype(str) == source
            for column, value in values.items():
                result.loc[mask, column] = value
    return result


def shuffle_action_identity(frame: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    result = frame.copy()
    columns = [column for column in ACTION_SPECIFIC_FEATURES if column in result.columns]
    for _, indices in result.groupby("group_id", sort=False).groups.items():
        idx = np.asarray(list(indices))
        if len(idx) < 2:
            continue
        donor = idx[rng.permutation(len(idx))]
        result.loc[idx, columns] = result.loc[donor, columns].to_numpy()
    return result


def wrong_trajectory(frame: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    result = frame.copy()
    targets = [
        "future_behavior_signal",
        "future_proximal_signal",
        "proximal_outcome_available",
    ]
    targets = [column for column in targets if column in result.columns]
    strata = ["stage", "course", "presentation", "action_family"]
    for _, indices in result.groupby(strata, sort=False).groups.items():
        idx = np.asarray(list(indices))
        if len(idx) < 2:
            continue
        donor = idx[rng.permutation(len(idx))]
        result.loc[idx, targets] = result.loc[donor, targets].to_numpy()
    return result


def time_reversal_targets(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    proxy = np.zeros(len(result), dtype=float)
    for family, indices in result.groupby("action_family", sort=False).groups.items():
        idx = np.asarray(list(indices))
        if family == "ASSESSMENT_COMPLETION":
            values = pd.to_numeric(result.loc[idx, "assessment_progress"], errors="coerce")
        elif family == "STUDY_REGULARITY":
            values = pd.to_numeric(result.loc[idx, "active_days"], errors="coerce") - pd.to_numeric(
                result.loc[idx, "inactive_streak"], errors="coerce"
            )
        elif family == "VLE_ENGAGEMENT":
            values = pd.to_numeric(result.loc[idx, "vle_intensity"], errors="coerce")
        elif family == "QUIZ_OR_RETRIEVAL_PRACTICE":
            values = pd.to_numeric(result.loc[idx, "opportunity_count"], errors="coerce")
        else:
            values = pd.to_numeric(result.loc[idx, "activity_trend"], errors="coerce")
        proxy[idx] = values.fillna(0.0).to_numpy(dtype=float)
    result["future_behavior_signal"] = proxy
    result["future_proximal_signal"] = np.nan
    result["proximal_outcome_available"] = 0
    return result


def fit_and_evaluate_control(
    raw_train: pd.DataFrame,
    raw_test: pd.DataFrame,
    family: str,
    parameters: dict[str, Any],
    control: str,
    seed: int,
) -> float:
    rng = np.random.default_rng(seed)
    raw_train = ordered(raw_train)
    raw_test = ordered(raw_test)

    original_labels = RelevanceTransformer(seed=seed)
    train = original_labels.fit_transform(raw_train)
    test = original_labels.transform(raw_test)

    if control == "NC1_LABEL_SHUFFLE_RETRAIN":
        train_features = train
        train_labels = shuffle_labels(train, rng)
        test_features = test
    elif control == "NC2A_TRAIN_STATE_SHUFFLE":
        train_features = shuffle_group_state(train, rng)
        train_labels = train
        test_features = test
    elif control == "NC2B_TEST_STATE_SHUFFLE":
        train_features = train
        train_labels = train
        test_features = shuffle_group_state(test, rng)
    elif control == "NC3_ACTION_IDENTITY_SHUFFLE_RETRAIN":
        train_features = shuffle_action_identity(train, rng)
        train_labels = train
        test_features = test
    elif control == "NC4_WRONG_TRAJECTORY_REBUILD":
        wrong_raw = wrong_trajectory(raw_train, rng)
        wrong_transformer = RelevanceTransformer(seed=seed)
        train_labels = wrong_transformer.fit_transform(wrong_raw)
        train_features = train_labels
        test_features = test
    elif control == "NC5_TIME_REVERSAL_PLACEBO":
        placebo_raw = time_reversal_targets(raw_train)
        placebo_transformer = RelevanceTransformer(seed=seed)
        train_labels = placebo_transformer.fit_transform(placebo_raw)
        train_features = train_labels
        test_features = test
    else:
        raise ValueError(f"Unknown control: {control}")

    preprocessor = FeaturePreprocessor(include_interactions=True)
    train_matrix = preprocessor.fit_transform(train_features)
    test_matrix = preprocessor.transform(test_features)
    ranker = fit_ranker(family, train_matrix, train_labels, parameters, seed)
    scored = test.copy()
    scored["control_score"] = predict_ranker(ranker, test_matrix)
    return float(aggregate_metrics(scored, "control_score")["ndcg_at_3"])


def batch_path(control: str, start: int, stop: int) -> Path:
    return CONTROL_OUT / "batches" / f"{control}__{start:04d}_{stop:04d}.csv"


def run_batch(
    raw: pd.DataFrame,
    control: str,
    start: int,
    stop: int,
) -> pd.DataFrame:
    rows = []
    for replicate in range(start, stop):
        fold_scores = []
        fold_weights = []
        for outer_fold in [0, 1, 2]:
            train = raw[raw["outer_fold"] != outer_fold].copy()
            test = raw[raw["outer_fold"] == outer_fold].copy()
            family, parameters = selected_model(outer_fold)
            score = fit_and_evaluate_control(
                train,
                test,
                family,
                parameters,
                control,
                SEED + replicate * 100 + outer_fold,
            )
            fold_scores.append(score)
            fold_weights.append(int(test["group_id"].nunique()))
        rows.append(
            {
                "control": control,
                "replicate": replicate,
                "ndcg_at_3": float(np.average(fold_scores, weights=fold_weights)),
            }
        )
    return pd.DataFrame(rows)


def summarize(registered_replicates: int) -> pd.DataFrame:
    real = json.loads((FINAL / "NESTED_OOF_RESULTS.json").read_text(encoding="utf-8"))[
        "metrics"
    ]["model_score"]["ndcg_at_3"]
    rows = []
    for control in CONTROLS:
        parts = sorted((CONTROL_OUT / "batches").glob(f"{control}__*.csv"))
        frame = pd.concat([pd.read_csv(path) for path in parts], ignore_index=True) if parts else pd.DataFrame()
        completed = int(frame["replicate"].nunique()) if len(frame) else 0
        if completed:
            null = frame.groupby("replicate", as_index=False)["ndcg_at_3"].mean()["ndcg_at_3"]
            p95 = float(np.quantile(null, 0.95))
            mean = float(np.mean(null))
            status = "PASS" if completed >= registered_replicates and real > p95 else (
                "FAIL" if completed >= registered_replicates else "PARTIAL"
            )
        else:
            p95 = mean = np.nan
            status = "NOT_RUN"
        rows.append(
            {
                "control": control,
                "real_ndcg_at_3": real,
                "null_mean": mean,
                "null_p95": p95,
                "registered_replicates": registered_replicates,
                "completed_replicates": completed,
                "status": status,
            }
        )
    summary = pd.DataFrame(rows)
    CONTROL_OUT.mkdir(parents=True, exist_ok=True)
    summary.to_csv(CONTROL_OUT / "SUMMARY.csv", index=False)
    atomic_json(
        CONTROL_OUT / "control_registry.json",
        {
            "status": "COMPLETE" if set(summary["status"]) <= {"PASS", "FAIL"} else "PARTIAL",
            "registered_replicates": registered_replicates,
            "controls": CONTROLS,
            "historical_null_score_file": "../NEGATIVE_CONTROLS.csv",
            "historical_null_score_status": "NULL_SCORE_SANITY_CHECK_NOT_RETRAINED_CONTROL",
        },
    )
    return summary


def update_progress(summary: pd.DataFrame) -> None:
    path = OUT / "PROGRESS.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    completed = int(summary["completed_replicates"].min()) if len(summary) else 0
    registered = int(summary["registered_replicates"].max()) if len(summary) else 0
    status = "COMPLETE" if completed >= registered else "PARTIAL"
    payload.setdefault("stages", {})["NEGATIVE_CONTROLS_RETRAINED"] = {
        "status": status,
        "completed_replicates_per_control": completed,
        "registered_replicates": registered,
    }
    atomic_json(path, payload)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--replicates", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=10)
    parser.add_argument("--control", choices=CONTROLS + ["all"], default="all")
    args = parser.parse_args()

    CONTROL_OUT.joinpath("batches").mkdir(parents=True, exist_ok=True)
    raw = pd.read_parquet(DATA / "candidate_rows.parquet")
    controls = CONTROLS if args.control == "all" else [args.control]
    for control in controls:
        for start in range(0, args.replicates, args.batch_size):
            stop = min(start + args.batch_size, args.replicates)
            path = batch_path(control, start, stop)
            if path.exists():
                continue
            frame = run_batch(raw, control, start, stop)
            temporary = path.with_suffix(".tmp.csv")
            frame.to_csv(temporary, index=False)
            os.replace(temporary, path)
    summary = summarize(args.replicates)
    update_progress(summary)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
