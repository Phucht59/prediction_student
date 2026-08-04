"""Run the complete preregistered V2.1 model/hyperparameter search.

The earlier corrected execution evaluated every model family but only the first
configuration of each family.  This runner preserves that evidence, clears only
the corrected final-OOF namespace, installs the complete preregistered Cartesian
grid, and then delegates to ``corrected_nested_evaluation``.  The operation is
resume-safe after the one-time archive step.
"""
from __future__ import annotations

import json
import os
import shutil
from pathlib import Path
from typing import Any

import corrected_nested_evaluation as evaluator
from scientific_core import hyperparameter_configs

OUT = evaluator.OUT
FINAL = evaluator.FINAL
MODEL_SELECTION = evaluator.MODEL_SELECTION
ARCHIVE = OUT / "pre_full_grid_archive"
MARKER = OUT / "FULL_REGISTERED_SEARCH.json"


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def full_candidate_grid(config: dict[str, Any]) -> list[tuple[str, dict[str, Any]]]:
    """Return every configuration frozen in the V2.1 protocol."""
    output: list[tuple[str, dict[str, Any]]] = []
    for family in config["models"]["candidates"]:
        frozen_space = config["models"].get("hyperparameters", {}).get(family, {})
        configurations = hyperparameter_configs(frozen_space)
        if not configurations:
            configurations = [{}]
        for parameters in configurations:
            registered = dict(parameters)
            registered.setdefault("n_jobs", 4)
            output.append((str(family), registered))
    return output


def expected_trial_count(config: dict[str, Any]) -> int:
    return len(full_candidate_grid(config))


def archive_first_configuration_execution() -> None:
    """Archive the first-config family screen once, without deleting evidence."""
    if MARKER.exists():
        return

    ARCHIVE.mkdir(parents=True, exist_ok=True)
    moves = [
        (FINAL, ARCHIVE / "final_oof_first_configuration"),
        (MODEL_SELECTION, ARCHIVE / "model_selection_first_configuration"),
    ]
    for source, destination in moves:
        if not source.exists():
            continue
        if destination.exists():
            raise RuntimeError(f"Archive destination already exists: {destination}")
        shutil.move(str(source), str(destination))

    atomic_json(
        MARKER,
        {
            "status": "RUNNING",
            "purpose": "complete_preregistered_hyperparameter_grid",
            "archived_preliminary_execution": True,
            "archive_directory": str(ARCHIVE.relative_to(OUT)).replace("\\", "/"),
        },
    )


def validate_completed_search(config: dict[str, Any]) -> None:
    expected = expected_trial_count(config)
    required_families = set(config["models"]["candidates"])
    fold_summaries: list[dict[str, Any]] = []
    for outer_fold in config["evaluation"]["outer_folds"]:
        trial_path = MODEL_SELECTION / f"fold_{int(outer_fold)}_trials.csv"
        selected_path = MODEL_SELECTION / f"fold_{int(outer_fold)}_selected.json"
        prediction_path = FINAL / f"fold_{int(outer_fold)}/predictions.parquet"
        for required in [trial_path, selected_path, prediction_path]:
            if not required.exists():
                raise RuntimeError(f"Full-grid execution is incomplete: {required}")

        import pandas as pd

        trials = pd.read_csv(trial_path)
        completed = trials[trials["status"] == "COMPLETE"]
        evaluated = set(completed["model"].astype(str))
        if len(trials) != expected:
            raise RuntimeError(
                f"Outer fold {outer_fold} has {len(trials)} trials; expected {expected}"
            )
        if evaluated != required_families:
            raise RuntimeError(
                f"Outer fold {outer_fold} model families {sorted(evaluated)} do not match "
                f"the frozen set {sorted(required_families)}"
            )
        selected = json.loads(selected_path.read_text(encoding="utf-8"))
        fold_summaries.append(
            {
                "outer_fold": int(outer_fold),
                "trial_count": int(len(trials)),
                "completed_trial_count": int(len(completed)),
                "selected_model": selected.get("model"),
                "selected_parameters": selected.get("parameters"),
            }
        )

    payload = json.loads(MARKER.read_text(encoding="utf-8"))
    payload.update(
        {
            "status": "COMPLETE",
            "expected_trials_per_outer_fold": expected,
            "folds": fold_summaries,
        }
    )
    atomic_json(MARKER, payload)


def main() -> None:
    config = evaluator.load_config()
    archive_first_configuration_execution()
    evaluator.candidate_grid = full_candidate_grid
    evaluator.main()
    validate_completed_search(config)
    print(MARKER.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
