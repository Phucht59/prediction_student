from __future__ import annotations

import importlib.util
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[3]
SCRIPT_DIR = ROOT / "scripts/recommend_hybrid/v2_1"
CONFIG = ROOT / "configs/recommend_hybrid/outcome_grounded_v2_1.yaml"


def load_module(filename: str, name: str):
    path = SCRIPT_DIR / filename
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_full_registered_grid_executes_every_frozen_configuration(monkeypatch):
    monkeypatch.syspath_prepend(str(SCRIPT_DIR))
    module = load_module("run_full_registered_search.py", "run_full_registered_search")
    config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    grid = module.full_candidate_grid(config)
    counts = {}
    for family, _ in grid:
        counts[family] = counts.get(family, 0) + 1
    assert counts == {
        "interaction_logistic": 3,
        "pairwise_logistic": 3,
        "lambdamart": 8,
        "boosted_tree": 4,
    }
    assert len(grid) == 18


def test_exact_control_runner_does_not_reduce_selected_tree_budget(monkeypatch):
    monkeypatch.syspath_prepend(str(SCRIPT_DIR))
    source = (SCRIPT_DIR / "run_exact_negative_controls.py").read_text(encoding="utf-8")
    assert 'min(int(control_parameters.get("n_estimators"' not in source
    assert "exact_parameters = dict(parameters)" in source


def test_exact_ablation_runner_does_not_reduce_selected_tree_budget():
    source = (SCRIPT_DIR / "run_exact_ablation.py").read_text(encoding="utf-8")
    assert 'min(int(ablation_parameters.get("n_estimators"' not in source
    assert "dict(parameters)" in source
