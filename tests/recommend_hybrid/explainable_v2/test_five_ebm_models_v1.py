from scripts.recommend_hybrid.explainable_v2 import train_five_ebm_models as runner


def test_five_action_models_are_locked():
    assert len(runner.ACTIONS) == 5
    assert len(set(runner.ACTIONS)) == 5


def test_ebm_interaction_budget_is_protocol_locked_and_interpretable():
    # Locked search allowed interactions in {0, 3, 5, 10}.
    # Panel-A selection chose 3 pairwise interactions under the frozen rule.
    assert runner.EBM_PARAMS["interactions"] == 3
    assert runner.EBM_PARAMS["interactions"] <= 3
    assert runner.LOCKED_GRID_SELECTED_CONFIG_ID == "a70599afad40"


def test_panel_a_only_contract():
    assert runner.EXPECTED_PANEL_A_CASES == 300
    assert runner.EXPECTED_ROWS == 1500


def test_ranking_metric_perfect_order_is_one():
    import numpy as np
    y = np.asarray([3.0, 2.0, 1.0, 0.0, 0.0])
    assert abs(runner._ndcg_at_k(y, y, 3) - 1.0) < 1e-12


def test_stage_is_merge_key_not_duplicated_suffix():
    import inspect
    source = inspect.getsource(runner._load_inputs)
    assert 'on=["query_id", "action_id", "stage"]' in source
    assert "stage_x" not in source
    assert "stage_y" not in source


def test_seed_disagreement_not_used_by_ebm_training():
    assert "seed_disagreement" not in runner.FEATURES


def test_ebm_feature_schema_has_expected_count():
    assert len(runner.FEATURES) == 16
    assert len(set(runner.FEATURES)) == 16


def test_locked_grid_selected_parameters_are_applied():
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


def test_locked_grid_selected_config_id_is_frozen():
    assert runner.LOCKED_GRID_SELECTED_CONFIG_ID == "a70599afad40"
