"""Isolated execution adapter for the restored, frozen UCI V5.1 protocol.

This module deliberately imports the archived V5.1 code instead of recreating its
model, preprocessing, split, loss, or fit procedures.
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent
HISTORICAL_ROOT = ROOT / "historical_uci_v5_1"
RUNTIME = ROOT / "runtime"
SEEDS = (42, 1201, 2026, 3407, 7319)


def historical_imports() -> None:
    """Make only the isolated historical package importable."""
    value = str(HISTORICAL_ROOT)
    if value not in sys.path:
        sys.path.insert(0, value)


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", delete=False, dir=path.parent) as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.flush()
        os.fsync(handle.fileno())
        temporary = Path(handle.name)
    temporary.replace(path)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def protocol_and_data(dataset: str):
    historical_imports()
    from src.studies.v5_1.common.protocol import load_protocol
    from src.studies.v5_1.common.uci_data import load_uci_v5_1

    protocol = load_protocol(dataset)
    # The archived V5.1 code remains self-contained, while raw input data stays
    # in the repository's canonical data location.  This avoids relying on the
    # local ``historical_uci_v5_1/data`` convenience junction.
    return protocol, load_uci_v5_1(ROOT.parents[1] / protocol["source"]["path"], dataset)


def fold_recipe(dataset: str, outer_fold: int) -> dict[str, Any]:
    """Load the exact frozen per-outer-fold selected recipe, without search."""
    recipes = json.loads((HISTORICAL_ROOT / "artifacts" / "v5_1" / dataset.replace("-", "_") / "selected_configs.json").read_text(encoding="utf-8"))
    recipe = dict(recipes[outer_fold])
    recipe["config"] = dict(recipe["config"])
    return recipe


def partition(dataset: str, outer_fold: int):
    historical_imports()
    from src.studies.v5_1.uci.runner import _outer_indices
    from src.studies.v5_1.common.uci_training import prepare_partition

    protocol, data = protocol_and_data(dataset)
    outer_train, outer_test = _outer_indices(protocol)[outer_fold]
    train, transformer = prepare_partition(data, outer_train, outer_train)
    test, _ = prepare_partition(data, outer_train, outer_test, fitted=transformer)
    return protocol, data, outer_train, outer_test, train, test, transformer


def source_partition_for_mat(data, target_test: np.ndarray, transformer, outer_fold: int):
    historical_imports()
    from sklearn.model_selection import StratifiedGroupKFold
    from src.studies.v5_1.uci.runner import _source_inputs
    from src.studies.v5_1.common.protocol import load_protocol
    from src.studies.v5_1.common.uci_data import load_uci_v5_1
    from src.studies.v5_1.common.uci_transfer import overlap_safe_source_indices

    protocol = load_protocol("student-por")
    source = load_uci_v5_1(ROOT.parents[1] / protocol["source"]["path"], "student-por")
    safe = overlap_safe_source_indices(source.quasi_groups, data.quasi_groups[target_test])
    splitter = StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=3407 + outer_fold)
    relative_train, relative_validation = next(splitter.split(safe, source.target[safe], source.quasi_groups[safe]))
    return _source_inputs(source, safe[relative_train], transformer), _source_inputs(source, safe[relative_validation], transformer), safe


def adapter_signature(dataset: str) -> dict[str, Any]:
    protocol, _ = protocol_and_data(dataset)
    return {
        "model_class": "SharedTrunkSubjectHeadsV51" if dataset == "student-mat" else "UCIHybridV51",
        "transfer_policy": "shared_trunk_subject_specific_heads" if dataset == "student-mat" else "standalone",
        "preprocessing": protocol["features"]["preprocessing"],
        "classes": ["low", "medium", "high"],
        "outer_folds": protocol["splits"]["outer_folds"],
        "seeds": list(SEEDS), "optimizer": "AdamW", "metric": "sklearn_macro_f1",
        "ensemble": "mean_probability_across_fixed_seeds",
        "recipes": [fold_recipe(dataset, fold) for fold in range(5)],
    }


def assert_equivalence(dataset: str) -> dict[str, Any]:
    signature = adapter_signature(dataset)
    # The recipe is itself the archived historical authority input. Explicit checks
    # prevent future adapter defaults from changing scientific fields.
    required = {"input_projection", "cnn_channels", "lstm_hidden", "lstm_layers", "context_hidden", "context_layers", "fusion_hidden", "dropout", "learning_rate", "weight_decay", "batch_size", "max_epochs", "patience"}
    for recipe in signature["recipes"]:
        missing = required - set(recipe["config"])
        if missing or recipe["fixed_epochs"] < 1:
            raise RuntimeError(f"Invalid archived recipe for {dataset}: {sorted(missing)}")
    if signature["seeds"] != list(SEEDS) or signature["outer_folds"] != 5:
        raise RuntimeError("Archived final evaluation matrix mismatch")
    return signature
