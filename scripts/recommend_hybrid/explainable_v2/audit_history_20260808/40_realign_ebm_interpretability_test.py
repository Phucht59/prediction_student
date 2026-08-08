from __future__ import annotations

import argparse
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", type=Path, required=True)
    args = ap.parse_args()
    root = args.repo.resolve()

    test_path = root / "tests/recommend_hybrid/explainable_v2/test_five_ebm_models_v1.py"
    runner_path = root / "scripts/recommend_hybrid/explainable_v2/train_five_ebm_models.py"

    for path in (test_path, runner_path):
        if not path.is_file():
            raise RuntimeError(f"MISSING_REQUIRED_FILE={path}")

    tests = test_path.read_text(encoding="utf-8")

    old = (
        "def test_ebm_is_additive_for_interpretability():\n"
        '    assert runner.EBM_PARAMS["interactions"] == 0\n'
    )
    new = (
        "def test_ebm_interaction_budget_is_protocol_locked_and_interpretable():\n"
        "    # Locked search allowed interactions in {0, 3, 5, 10}.\n"
        "    # Panel-A selection chose 3 pairwise interactions under the frozen rule.\n"
        '    assert runner.EBM_PARAMS["interactions"] == 3\n'
        '    assert runner.EBM_PARAMS["interactions"] <= 3\n'
        '    assert runner.LOCKED_GRID_SELECTED_CONFIG_ID == "a70599afad40"\n'
    )

    if old not in tests:
        raise RuntimeError(
            "OUTDATED_ADDITIVE_TEST_BLOCK_NOT_FOUND; inspect test file before modifying"
        )

    tests = tests.replace(old, new, 1)
    test_path.write_text(tests, encoding="utf-8")

    runner = runner_path.read_text(encoding="utf-8")
    changed = runner.replace("additive EBM", "EBM with a protocol-selected limited interaction budget")
    changed = changed.replace("additive ExplainableBoostingRegressor", "ExplainableBoostingRegressor with a protocol-selected limited interaction budget")
    if changed != runner:
        runner_path.write_text(changed, encoding="utf-8")

    print("=== EBM INTERPRETABILITY TEST REALIGNMENT ===")
    print("OLD_ASSUMPTION=PURE_ADDITIVE_INTERACTIONS_0")
    print("LOCKED_SELECTED_INTERACTIONS=3")
    print("LOCKED_SELECTED_CONFIG_ID=a70599afad40")
    print("INTERPRETABILITY_CLAIM=LIMITED_PAIRWISE_INTERACTION_EBM")
    print("MODEL_PARAMETERS_CHANGED=FALSE")
    print("DATA_ARTIFACTS_MODIFIED=FALSE")
    print("PANEL_B_TOUCHED=FALSE")
    print("RUNTIME_AUTHORIZED=FALSE")
    print("TEST_PROTOCOL_REALIGNMENT=PASS")
    print("NEXT_ACTION=RERUN_TESTS_THEN_RETRAIN_FINAL_5_EBM")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
