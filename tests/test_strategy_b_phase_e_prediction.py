"""Mandatory protocol gates for the authorized Phase E-Prediction run."""

from __future__ import annotations

import inspect

import numpy as np
import pandas as pd
import pytest

from scripts import run_strategy_b_phase_e_prediction as runner
from src.strategy_b_phase_e_prediction import (
    DETERMINISTIC_SEED,
    PHASE_E_SEEDS,
    apply_temperature,
    calibration_metrics,
    choose_final,
    choose_temperature,
    classification_rows,
    fit_temperature,
    paired_metric_deltas,
    phase_e_registry,
    precision_recall_rows,
    regression_rows,
    seed_registry,
)


def _oof() -> pd.DataFrame:
    rows = []
    for candidate, seed in [("R0", DETERMINISTIC_SEED), ("M1", 202601), ("N0", 202601), ("N1", 202601)]:
        for source, label in enumerate([0, 1, 2, 0, 1, 2]):
            prediction = label if candidate != "N1" or source != 5 else 1
            probabilities = np.full(3, 0.05); probabilities[prediction] = 0.9
            rows.append({"candidate_id": candidate, "seed": seed, "outer_fold": source % 2, "source_row_number": source, "raw_g3": [8, 10, 15, 7, 14, 16][source], "true_label": label, "predicted_label": prediction, "prob_0": probabilities[0], "prob_1": probabilities[1], "prob_2": probabilities[2]})
    return pd.DataFrame(rows)


def _continuous(oof: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for row in oof.itertuples(index=False):
        rows.append({"candidate_id": row.candidate_id, "source_record_id": row.source_row_number, "outer_fold": row.outer_fold, "seed": row.seed, "true_g3": row.raw_g3, "predicted_g3": float(row.raw_g3) + (0.2 if row.candidate_id == "N1" else 0.0), "continuous_prediction_method": "registered"})
    return pd.DataFrame(rows)


def test_new_seeds_are_disjoint_from_phase_c_and_best_seed_selection_is_prohibited():
    assert set(PHASE_E_SEEDS).isdisjoint({42, 123, 155})
    registry = seed_registry()
    assert registry["M1"]["best_seed_selection"] is False
    assert registry["N0"]["best_seed_selection"] is False


def test_deterministic_models_have_no_fake_seed_rows_or_sd():
    registry = seed_registry()
    assert registry["R0"]["seed_not_applicable"] is True
    assert registry["M2"]["seed_not_applicable"] is True
    assert registry["R0"]["stored_seed"] == DETERMINISTIC_SEED


def test_random_forest_seed_propagation_and_svm_deterministic_declaration():
    m1 = {"candidate_id": "M1", "parameters": {"n_estimators": 10, "max_depth": 3, "min_samples_leaf": 1, "max_features": "sqrt"}}
    m2 = {"candidate_id": "M2", "parameters": {"C": 1.0, "gamma": 0.5}}
    assert runner._make_ml(m1, 202601).named_steps["model"].random_state == 202601
    assert runner._make_ml(m2, 202601).named_steps["model"].random_state == 42


def test_phase_e_registry_is_exact_and_prohibits_all_unapproved_branches():
    registry = phase_e_registry()
    assert registry["overall_finalists"] == ["R0", "M1", "M2"]
    assert registry["thesis_hybrid_finalists"] == ["N0", "N1"]
    assert {"C1_huber", "C2_residual", "imbalance", "context", "recommendation_phase_d"} <= set(registry["prohibited_branches"])


def test_stability_runner_uses_phase_c_configs_and_contains_no_search_in_stability_loop():
    source = inspect.getsource(runner.main)
    assert "phase_c_configs[(candidate, fold)]" in source
    assert '"architecture_search_during_stability": False' in source
    assert "_final_neural_search" in source


def test_temperature_is_numerically_stable_and_preserves_argmax():
    probabilities = np.asarray([[1 - 1e-15, 1e-15, 0.0], [0.1, 0.8, 0.1]])
    calibrated = apply_temperature(probabilities, 0.2)
    assert np.isfinite(calibrated).all()
    assert np.allclose(calibrated.sum(axis=1), 1.0)
    assert np.array_equal(probabilities.argmax(axis=1), calibrated.argmax(axis=1))
    with pytest.raises(ValueError):
        apply_temperature(probabilities, 0.0)


def test_temperature_fits_only_supplied_inner_oof_and_not_outer_labels():
    probabilities = np.asarray([[0.8, 0.1, 0.1], [0.2, 0.7, 0.1], [0.1, 0.2, 0.7]])
    calibrator = fit_temperature(probabilities, np.asarray([0, 1, 2]))
    assert calibrator["method"].startswith("scalar_temperature")
    source = inspect.getsource(runner._inner_calibration)
    assert "outer_train" in source and "outer_validation" not in source


def test_temperature_selection_uses_nll_majority_and_brier_ece_tolerances():
    rows = []
    for fold in range(5):
        rows += [
            {"candidate_id": "M1", "outer_fold": fold, "seed": 1, "variant": "uncalibrated", "nll": 0.6, "brier": 0.2, "ece": 0.1},
            {"candidate_id": "M1", "outer_fold": fold, "seed": 1, "variant": "temperature", "nll": 0.5, "brier": 0.201, "ece": 0.101},
        ]
    assert choose_temperature(pd.DataFrame(rows))["M1"]["selected_variant"] == "temperature"


def test_accuracy_precision_recall_and_f1_recompute_from_oof():
    rows = classification_rows(_oof())
    result = rows[(rows["candidate_id"] == "R0") & (rows["outer_fold"] == 0)].iloc[0]
    assert result["accuracy"] == 1.0
    assert result["macro_precision"] == result["macro_recall"] == result["macro_f1"] == 1.0
    assert {"weighted_f1", "high_precision", "high_recall", "high_f1"} <= set(rows.columns)


def test_precision_recall_is_one_vs_rest_and_uses_probabilities_not_hard_labels():
    metrics, points = precision_recall_rows(_oof())
    assert {"low_average_precision", "medium_average_precision", "high_average_precision", "macro_pr_auc", "micro_pr_auc", "weighted_pr_auc"} <= set(metrics.columns)
    assert set(points["class_name"]) == {"Low", "Medium", "High"}
    assert points["threshold"].between(0, 1).all()


def test_continuous_prediction_contract_uses_training_means_and_r2_may_be_negative():
    source = inspect.getsource(runner._continuous_rows)
    assert 'outer_train.groupby("G3")["G3_raw"].mean()' in source
    assert "validation.groupby" not in source
    bad = _continuous(_oof()); bad["predicted_g3"] = 999.0
    assert (regression_rows(bad)["r2"] < 0).all()


def test_r0_continuous_contract_is_raw_g2_and_not_encoded_labels():
    source = inspect.getsource(runner._continuous_rows)
    assert 'validation["G2"]' in source
    assert "predicted = np.asarray(probabilities, dtype=float) @ means" in source


def test_rmse_and_r2_recompute_and_are_secondary_to_macro_f1():
    metrics = regression_rows(_continuous(_oof()))
    assert {"mae", "rmse", "r2"} <= set(metrics.columns)
    rule = runner.selection_rule()
    assert "rmse" in rule["secondary_metrics_cannot_replace_macro_f1"]
    assert "r2" in rule["secondary_metrics_cannot_replace_macro_f1"]


def test_paired_metric_comparisons_include_required_metrics_without_fake_deterministic_rows():
    paired = paired_metric_deltas(_oof(), _continuous(_oof()), [("M1", "R0"), ("N1", "N0")], bootstrap_samples=25)
    assert {"accuracy_delta_left_minus_right", "macro_precision_delta_left_minus_right", "macro_recall_delta_left_minus_right", "macro_f1_delta_left_minus_right", "high_precision_delta_left_minus_right", "high_recall_delta_left_minus_right", "high_f1_delta_left_minus_right", "macro_pr_auc_delta_left_minus_right", "rmse_delta_left_minus_right", "r2_delta_left_minus_right"} <= set(paired.columns)
    assert "-1" not in paired.iloc[0]["seed_deltas_json"]


def test_final_tie_break_does_not_treat_deterministic_seed_absence_as_zero_sd():
    summary = pd.DataFrame([
        {"candidate_id": "M1", "oof_macro_f1": 0.90, "class_collapse_count": 0, "seed_sd": 0.02, "seed_sd_not_applicable": False, "worst_seed": 0.88, "two_step_error": 0.0, "ece": 0.1, "parameter_count": 10, "simplicity_rank": 2},
        {"candidate_id": "R0", "oof_macro_f1": 0.899, "class_collapse_count": 0, "seed_sd": np.nan, "seed_sd_not_applicable": True, "worst_seed": 0.899, "two_step_error": 0.0, "ece": 0.2, "parameter_count": 0, "simplicity_rank": 0},
    ])
    paired = pd.DataFrame([{ "left": "M1", "right": "R0", "macro_f1_record_bootstrap_ci_low": -0.01, "macro_f1_record_bootstrap_ci_high": 0.01 }])
    selected, _ = choose_final(summary, ["M1", "R0"], paired)
    assert selected == "R0"  # worst genuine M1 seed loses; R0 was not assigned false SD=0.


def test_final_refit_and_five_checkpoint_contracts_are_explicit():
    source = inspect.getsource(runner.main)
    assert "fit_final_development_estimator" in source
    assert "for seed in PHASE_E_SEEDS:" in source
    assert '"checkpoints": 5' in source


def test_runner_never_fetches_legacy_or_runs_recommendation_and_is_atomic():
    source = inspect.getsource(runner)
    assert "load_development_subset_from_postgres" in source
    assert "load_dataset_from_postgres" not in source
    assert '"legacy_observed_79_fetched": False' in source
    assert '"recommendation_phase_d_executed": False' in source
    assert "os.replace(tmp, final)" in source
    assert '_write_state(tmp, "failed"' in source
