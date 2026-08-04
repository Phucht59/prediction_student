"""Run exact-hyperparameter retrained negative controls for V2.1.

The first control implementation reduced LambdaMART from the selected 100 trees
to 10 trees.  That changes model capacity and makes the null comparison too easy.
This runner archives any reduced-budget batches once, monkey-patches the control
fit routine to use the exact selected outer-fold configuration, and delegates to
the resumable registered control runner.
"""
from __future__ import annotations

import json
import os
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

import corrected_negative_controls as controls
from scientific_core import (
    FeaturePreprocessor,
    RelevanceTransformer,
    aggregate_metrics,
    fit_ranker,
    predict_ranker,
)

OUT = controls.OUT
CONTROL_OUT = controls.CONTROL_OUT
ARCHIVE = OUT / "negative_controls_reduced_budget_archive"
MARKER = OUT / "EXACT_NEGATIVE_CONTROL_EXECUTION.json"


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def archive_reduced_budget_controls_once() -> None:
    if MARKER.exists():
        return
    if CONTROL_OUT.exists():
        if ARCHIVE.exists():
            raise RuntimeError(f"Control archive already exists: {ARCHIVE}")
        shutil.move(str(CONTROL_OUT), str(ARCHIVE))
    atomic_json(
        MARKER,
        {
            "status": "RUNNING",
            "selected_hyperparameters_required": True,
            "reduced_tree_budget_allowed": False,
            "archived_reduced_budget_outputs": ARCHIVE.exists(),
        },
    )


def exact_fit_and_evaluate_control(
    raw_train: pd.DataFrame,
    raw_test: pd.DataFrame,
    family: str,
    parameters: dict[str, Any],
    control: str,
    seed: int,
) -> float:
    """Refit a control using exactly the selected model family/configuration."""
    rng = np.random.default_rng(seed)
    raw_train = controls.ordered(raw_train)
    raw_test = controls.ordered(raw_test)

    original_labels = RelevanceTransformer(seed=seed)
    train = original_labels.fit_transform(raw_train)
    test = original_labels.transform(raw_test)

    if control == "NC1_LABEL_SHUFFLE_RETRAIN":
        train_features = train
        train_labels = controls.shuffle_labels(train, rng)
        test_features = test
    elif control == "NC2A_TRAIN_STATE_SHUFFLE":
        train_features = controls.shuffle_group_state(train, rng)
        train_labels = train
        test_features = test
    elif control == "NC2B_TEST_STATE_SHUFFLE":
        train_features = train
        train_labels = train
        test_features = controls.shuffle_group_state(test, rng)
    elif control == "NC3_ACTION_IDENTITY_SHUFFLE_RETRAIN":
        train_features = controls.shuffle_action_identity(train, rng)
        train_labels = train
        test_features = test
    elif control == "NC4_WRONG_TRAJECTORY_REBUILD":
        wrong_raw = controls.wrong_trajectory(raw_train, rng)
        wrong_transformer = RelevanceTransformer(seed=seed)
        train_labels = wrong_transformer.fit_transform(wrong_raw)
        train_features = train_labels
        test_features = test
    elif control == "NC5_TIME_REVERSAL_PLACEBO":
        placebo_raw = controls.time_reversal_targets(raw_train)
        placebo_transformer = RelevanceTransformer(seed=seed)
        train_labels = placebo_transformer.fit_transform(placebo_raw)
        train_features = train_labels
        test_features = test
    else:
        raise ValueError(f"Unknown control: {control}")

    preprocessor = FeaturePreprocessor(include_interactions=True)
    train_matrix = preprocessor.fit_transform(train_features)
    test_matrix = preprocessor.transform(test_features)
    exact_parameters = dict(parameters)
    ranker = fit_ranker(family, train_matrix, train_labels, exact_parameters, seed)
    scored = test.copy()
    scored["control_score"] = predict_ranker(ranker, test_matrix)
    return float(aggregate_metrics(scored, "control_score")["ndcg_at_3"])


def finalize_marker() -> None:
    summary_path = CONTROL_OUT / "SUMMARY.csv"
    if not summary_path.exists():
        return
    summary = pd.read_csv(summary_path)
    complete = bool(len(summary)) and bool(
        (summary["completed_replicates"] >= summary["registered_replicates"]).all()
    )
    payload = json.loads(MARKER.read_text(encoding="utf-8"))
    payload.update(
        {
            "status": "COMPLETE" if complete else "PARTIAL",
            "controls": summary.to_dict(orient="records"),
        }
    )
    atomic_json(MARKER, payload)


def main() -> None:
    archive_reduced_budget_controls_once()
    controls.fit_and_evaluate_control = exact_fit_and_evaluate_control
    controls.main()
    finalize_marker()


if __name__ == "__main__":
    main()
