from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[3]
SCRIPT = ROOT / "scripts/recommend_hybrid/v2_1"
CONFIG = ROOT / "configs/recommend_hybrid/outcome_grounded_v2_1.yaml"


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_registered_grid_has_all_frozen_configurations(monkeypatch):
    monkeypatch.syspath_prepend(str(SCRIPT))
    module = load_module("run_full_registered_search_test", SCRIPT / "run_full_registered_search.py")
    config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    grid = module.full_candidate_grid(config)
    assert len(grid) == 18
    assert {family for family, _ in grid} == {
        "interaction_logistic",
        "pairwise_logistic",
        "lambdamart",
        "boosted_tree",
    }


def test_resumable_runner_checkpoints_trial_and_inner_fold():
    text = (SCRIPT / "run_resumable_full_registered_search.py").read_text(encoding="utf-8")
    assert 'f"inner_{inner_fold}.json"' in text
    assert 'f"trial_{trial_number:03d}"' in text
    assert '"TRIAL_AND_INNER_FOLD_RESUMABLE"' in text
    assert "--trial-start" in text
    assert "--trial-stop" in text


def test_postsearch_evidence_is_bound_to_selected_model_authority():
    helper = (SCRIPT / "postsearch_authority.py").read_text(encoding="utf-8")
    controls = (SCRIPT / "run_authority_bound_negative_controls.py").read_text(encoding="utf-8")
    ablations = (SCRIPT / "run_authority_bound_ablation.py").read_text(encoding="utf-8")
    release = (SCRIPT / "run_authority_bound_release.py").read_text(encoding="utf-8")
    assert "model_authority_sha256" in helper
    assert "prepare_namespace" in controls
    assert "prepare_namespace" in ablations
    assert "assert_bound" in release


def test_current_partial_control_registry_is_not_full_scientific_evidence():
    path = ROOT / "artifacts/recommend_hybrid/outcome_grounded_v2_1/negative_controls_retrained/control_registry.json"
    if not path.exists():
        return
    payload = json.loads(path.read_text(encoding="utf-8"))
    registered = int(payload.get("registered_replicates", 0))
    assert registered < 200 or payload.get("status") != "COMPLETE", (
        "A control registry may be COMPLETE for the final gate only after the "
        "registered 200-replicate authority-bound execution."
    )
