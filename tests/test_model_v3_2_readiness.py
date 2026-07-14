import json
import pandas as pd
import pytest

from src.evaluation.model_v3_2 import (
    V3_2_PROTOCOL_VERSION, build_b0_selection_contract, build_shared_inner_split_manifest,
    round_half_up_median, validate_authorization, validate_inner_split_manifest,
    validate_pooled_oof_exact, validate_selected_trials,
)
from src.evaluation.model_v3_protocol import checksum


def _records():
    return {0: pd.DataFrame({"record_id": [f"r{i}" for i in range(12)], "true_label": [0, 1, 2] * 4})}


def test_shared_inner_split_is_model_independent_and_excludes_outer_validation():
    manifest = build_shared_inner_split_manifest(_records(), "fold")
    errors = validate_inner_split_manifest(manifest, {0: set(_records()[0].record_id)}, {0: {"outer"}}, set())
    assert not any(errors.values())
    mutated = json.loads(json.dumps(manifest)); mutated["assignments"][0]["record_id"] = "outer"
    assert validate_inner_split_manifest(mutated, {0: set(_records()[0].record_id)}, {0: {"outer"}}, set())["outer_validation_present"] > 0


def test_b0_contract_has_ten_studies_and_four_alpha_trials():
    features = {x: {"semantic_checksum": x} for x in ["late_stage", "early_warning"]}
    c = build_b0_selection_contract("run", "commit", "inner", features)
    assert len(c["studies"]) == 10 and all(x["trial_budget"] == 4 and x["expected_inner_evaluations"] == 12 for x in c["studies"])


def test_refit_epoch_rounding_rule():
    assert round_half_up_median([2, 3, 4], 60) == 3
    assert round_half_up_median([2, 3, 3], 60) == 3
    with pytest.raises(ValueError): round_half_up_median([0, 1, 2], 60)


def _selection(study, b0=False):
    rows = []
    for trial in range(study["trial_budget"]):
        payload = {"alpha": [.01, .1, 1., 10.][trial]} if b0 else {"hidden_width": 8 + trial}
        for inner in range(3):
            rows.append({"study_id": study["study_id"], "trial_id": trial, "inner_fold": inner, "status": "completed", "config_payload": json.dumps(payload), "config_checksum": checksum(payload), "macro_f1": .1 + trial / 10, "ordinal_mae": .5, "rmse_raw": 4 - trial / 10, "mae_raw": 3 - trial / 10})
    chosen = study["trial_budget"] - 1
    payload = json.loads(rows[-1]["config_payload"])
    return pd.DataFrame(rows), pd.DataFrame([{ "study_id": study["study_id"], "selected_trial_id": chosen, "config_payload": json.dumps(payload), "config_checksum": checksum(payload), "inner_split_manifest_checksum": "inner"}])


def test_selection_validator_rejects_nonbest_missing_and_duplicate_selected():
    study = {"study_id": "s", "model_family": "M0", "trial_budget": 2}
    trials, selected = _selection(study)
    assert not any(validate_selected_trials([study], trials, selected, "inner", {}).values())
    selected.loc[0, "selected_trial_id"] = 0
    assert validate_selected_trials([study], trials, selected, "inner", {})["selected_not_best"] == 1
    assert validate_selected_trials([study], trials.iloc[:-1], selected, "inner", {})["bad_inner_folds"] > 0
    assert validate_selected_trials([study], trials, pd.DataFrame(), "inner", {})["missing_selected"] == 1


def test_b0_best_alpha_and_pooled_exact_coverage():
    study = {"study_id": "b", "model_family": "B0", "trial_budget": 4}
    trials, selected = _selection(study, True)
    assert not any(validate_selected_trials([study], trials, selected, "inner", {}).values())
    pred = pd.DataFrame({"model_family": ["B0"] * 3, "track": ["late"] * 3, "training_seed": [0] * 3, "record_id": ["a", "b", "c"], "raw_g3": [1., 2., 3.], "predicted_g3_raw": [1., 2., 3.]})
    validate_pooled_oof_exact(pred, {"a", "b", "c"})
    with pytest.raises(ValueError): validate_pooled_oof_exact(pred.iloc[:-1], {"a", "b", "c"})


def test_authorization_rejects_false_smoke_mismatch_and_counts():
    expected = {"run_id": "run", "semantic_checksum": "e", "jobs": [{"track": "late_stage", "outer_fold": i, "expected_record_count": 1, "smoke": False} for i in range(5)]}
    empty = {"run_id": "run", "semantic_checksum": "x"}
    auth = {"execution_mode": "full", "compute_authorized": False, "run_id": "run", "source_commit": "c", "source_tree_clean": True}
    with pytest.raises(ValueError): validate_authorization(auth, expected, empty, empty, empty, empty, empty, empty, empty, empty, source_commit="c", tree_clean=True)
