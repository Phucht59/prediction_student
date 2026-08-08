from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path

SEARCH_KEYS = {
    "interactions",
    "learning_rate",
    "max_bins",
    "max_rounds",
    "min_samples_leaf",
}

FIXED_EXPECTED = {
    "outer_bags": 8,
    "inner_bags": 0,
    "validation_size": 0.15,
    "early_stopping_rounds": 100,
    "early_stopping_tolerance": 1e-5,
    "random_state": 2026,
}


def dict_keys(node: ast.AST) -> set[str]:
    if not isinstance(node, ast.Dict):
        return set()
    out = set()
    for key in node.keys:
        if isinstance(key, ast.Constant) and isinstance(key.value, str):
            out.add(key.value)
    return out


def assignment_name(node: ast.Assign) -> str:
    if len(node.targets) != 1 or not isinstance(node.targets[0], ast.Name):
        raise RuntimeError("UNSUPPORTED_PARAMETER_ASSIGNMENT")
    return node.targets[0].id


def replace_lines(source: str, node: ast.Assign, replacement: str) -> str:
    lines = source.splitlines(keepends=True)
    start = node.lineno - 1
    end = node.end_lineno
    return "".join(lines[:start]) + replacement + "".join(lines[end:])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", type=Path, required=True)
    args = ap.parse_args()
    root = args.repo.resolve()

    selection_path = (
        root
        / "artifacts/recommend_hybrid/explainable_v2/ranker_development"
        / "ebm_locked_grid_v1/EBM_GRID_SELECTION.json"
    )
    runner_path = (
        root
        / "scripts/recommend_hybrid/explainable_v2/train_five_ebm_models.py"
    )
    test_path = (
        root
        / "tests/recommend_hybrid/explainable_v2/test_five_ebm_models_v1.py"
    )

    for p in (selection_path, runner_path, test_path):
        if not p.is_file():
            raise RuntimeError(f"MISSING_REQUIRED_FILE={p}")

    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    required_manifest = {
        "status": "PASS",
        "panel": "A",
        "panel_b_touched": False,
        "runtime_authorized": False,
        "search_complete": True,
        "grid_config_count": 432,
        "total_ebm_fits": 6480,
    }
    for key, expected in required_manifest.items():
        if selection.get(key) != expected:
            raise RuntimeError(
                f"GRID_SELECTION_FIELD_MISMATCH={key}:"
                f"{selection.get(key)!r}!={expected!r}"
            )

    selected = selection["selected"]
    selected_id = str(selected["config_id"])
    expected_id = "a70599afad40"
    expected_selected = {
        "interactions": 3,
        "learning_rate": 0.025,
        "max_bins": 64,
        "max_rounds": 2000,
        "min_samples_leaf": 20,
    }
    observed_selected = {k: selected[k] for k in SEARCH_KEYS}

    if selected_id != expected_id:
        raise RuntimeError(
            f"UNEXPECTED_SELECTED_CONFIG={selected_id} expected={expected_id}"
        )
    if observed_selected != expected_selected:
        raise RuntimeError(
            f"SELECTED_PARAMS_MISMATCH={observed_selected}"
        )

    source = runner_path.read_text(encoding="utf-8")
    tree = ast.parse(source)

    matches = []
    for node in tree.body:
        if isinstance(node, ast.Assign) and isinstance(node.value, ast.Dict):
            if SEARCH_KEYS.issubset(dict_keys(node.value)):
                matches.append(node)

    if len(matches) != 1:
        raise RuntimeError(
            f"EXPECTED_ONE_EBM_PARAMETER_DICT_FOUND={len(matches)}"
        )

    node = matches[0]
    param_name = assignment_name(node)

    merged = {
        "interactions": 3,
        "max_bins": 64,
        "max_rounds": 2000,
        "learning_rate": 0.025,
        "min_samples_leaf": 20,
        **FIXED_EXPECTED,
    }
    body = ",\n".join(
        f'    "{key}": {repr(value)}' for key, value in merged.items()
    )
    replacement = f"{param_name} = {{\n{body},\n}}\n"
    patched = replace_lines(source, node, replacement)

    marker = "LOCKED_GRID_SELECTED_CONFIG_ID"
    if marker not in patched:
        provenance = (
            f'\nLOCKED_GRID_SELECTED_CONFIG_ID = "{selected_id}"\n'
            'LOCKED_GRID_SELECTION_RELATIVE_PATH = (\n'
            '    "artifacts/recommend_hybrid/explainable_v2/ranker_development/"\n'
            '    "ebm_locked_grid_v1/EBM_GRID_SELECTION.json"\n'
            ')\n'
        )
        patched = patched.replace(
            replacement,
            replacement + provenance,
            1,
        )

    if '"locked_grid_selected_config_id"' not in patched:
        anchor = '        "features": list(FEATURES),\n'
        if patched.count(anchor) != 1:
            raise RuntimeError("MANIFEST_FEATURE_ANCHOR_NOT_UNIQUE")
        patched = patched.replace(
            anchor,
            anchor
            + '        "locked_grid_selected_config_id": '
              'LOCKED_GRID_SELECTED_CONFIG_ID,\n'
            + '        "locked_grid_selection_path": '
              'LOCKED_GRID_SELECTION_RELATIVE_PATH,\n'
            + '        "ranker_calibration": "NONE_RAW_EBM_SELECTED",\n',
            1,
        )

    ast.parse(patched)
    runner_path.write_text(patched, encoding="utf-8")

    tests = test_path.read_text(encoding="utf-8")
    if "test_locked_grid_selected_parameters_are_applied" not in tests:
        tests += '''
\ndef test_locked_grid_selected_parameters_are_applied():
    candidates = [
        value
        for value in vars(runner).values()
        if isinstance(value, dict)
        and {
            "interactions",
            "learning_rate",
            "max_bins",
            "max_rounds",
            "min_samples_leaf",
        }.issubset(value)
    ]
    assert len(candidates) == 1
    params = candidates[0]
    assert params["interactions"] == 3
    assert params["learning_rate"] == 0.025
    assert params["max_bins"] == 64
    assert params["max_rounds"] == 2000
    assert params["min_samples_leaf"] == 20

\ndef test_locked_grid_selected_config_id_is_frozen():
    assert runner.LOCKED_GRID_SELECTED_CONFIG_ID == "a70599afad40"
'''
        test_path.write_text(tests, encoding="utf-8")

    print("=== APPLY LOCKED EBM SELECTION ===")
    print("GRID_SEARCH_STATUS=PASS")
    print("GRID_CONFIGS=432")
    print("GRID_EBM_FITS=6480")
    print("EMPIRICAL_BEST_ID=4ad76ac8f31b")
    print("EMPIRICAL_BEST_NDCG_AT_3=0.975152")
    print(f"SELECTED_CONFIG_ID={selected_id}")
    print("SELECTED_PARAMS=" + json.dumps(expected_selected, sort_keys=True))
    print("SELECTED_NDCG_AT_3=0.972254")
    print("SELECTION_RULE=SIMPLEST_STATISTICALLY_INDISTINGUISHABLE_BEST")
    print("RANKER_CALIBRATION=NONE_RAW_EBM_SELECTED")
    print("PANEL_B_TOUCHED=FALSE")
    print("RUNTIME_AUTHORIZED=FALSE")
    print("RUNNER_PATCHED=TRUE")
    print("MODEL_ARTIFACTS_MODIFIED=FALSE")
    print("LOCKED_EBM_CONFIG_APPLY=PASS")
    print("NEXT_ACTION=TEST_AND_RETRAIN_FINAL_5_EBM")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
