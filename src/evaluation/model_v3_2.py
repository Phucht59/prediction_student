"""V3.2 execution-readiness contracts and strict validators (no model training)."""
from __future__ import annotations

import math
import subprocess
from decimal import Decimal, ROUND_HALF_UP
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

from src.evaluation.model_v3_protocol import (
    MODEL_REGISTRY, SEEDS, V3_1_PROTOCOL_VERSION, checksum, regression_metric_summary,
)

V3_2_PROTOCOL_VERSION = "model_v3_2"


def git_tree_clean(root: str) -> bool:
    return not subprocess.check_output(["git", "status", "--porcelain", "--untracked-files=no"], cwd=root, text=True).strip()


def inner_split_seed(outer_fold: int, fold_checksum: str) -> int:
    return int(checksum({"protocol": V3_2_PROTOCOL_VERSION, "outer_fold": outer_fold, "fold_checksum": fold_checksum})[:8], 16) % (2**31 - 1)


def build_shared_inner_split_manifest(outer_train_by_fold: dict[int, pd.DataFrame], fold_checksum: str) -> dict[str, Any]:
    assignments = []
    for outer_fold, records in outer_train_by_fold.items():
        seed = inner_split_seed(outer_fold, fold_checksum)
        labels = records["true_label"].to_numpy(int)
        split = StratifiedKFold(n_splits=3, shuffle=True, random_state=seed)
        for inner_fold, (train_idx, valid_idx) in enumerate(split.split(records, labels)):
            for index in train_idx:
                row = records.iloc[index]
                assignments.append({"outer_fold": outer_fold, "inner_fold": inner_fold, "record_id": row.record_id,
                                    "source_record_identity": row.record_id, "role": "train", "true_class": int(row.true_label)})
            for index in valid_idx:
                row = records.iloc[index]
                assignments.append({"outer_fold": outer_fold, "inner_fold": inner_fold, "record_id": row.record_id,
                                    "source_record_identity": row.record_id, "role": "validation", "true_class": int(row.true_label)})
    payload = {"contract_version": V3_2_PROTOCOL_VERSION, "created_before_compute": True,
               "fold_manifest_checksum": fold_checksum, "inner_folds": 3,
               "assignments": assignments}
    payload["semantic_checksum"] = checksum(payload)
    return payload


def validate_inner_split_manifest(payload: dict[str, Any], outer_train_ids: dict[int, set[str]],
                                  outer_validation_ids: dict[int, set[str]], legacy_ids: set[str]) -> dict[str, int]:
    rows = pd.DataFrame(payload.get("assignments", []))
    errors = {"missing_outer_train": 0, "outer_validation_present": 0, "legacy_present": 0,
              "overlap": 0, "validation_not_once": 0, "bad_fold_count": 0}
    for outer_fold, expected in outer_train_ids.items():
        subset = rows[rows.outer_fold == outer_fold]
        if subset.empty or subset.inner_fold.nunique() != 3:
            errors["bad_fold_count"] += 1
            continue
        for inner_fold in range(3):
            current = subset[subset.inner_fold == inner_fold]
            train_ids = set(current[current.role == "train"].record_id)
            valid_ids = set(current[current.role == "validation"].record_id)
            errors["overlap"] += int(bool(train_ids & valid_ids))
            errors["missing_outer_train"] += int(train_ids | valid_ids != expected)
        valid_counts = subset[subset.role == "validation"].groupby("record_id").size()
        errors["validation_not_once"] += int(not (set(valid_counts.index) == expected and (valid_counts == 1).all()))
        errors["outer_validation_present"] += len(set(subset.record_id) & outer_validation_ids[outer_fold])
        errors["legacy_present"] += len(set(subset.record_id) & legacy_ids)
    return errors


def round_half_up_median(epochs: list[int], max_epochs: int) -> int:
    if len(epochs) != 3:
        raise ValueError("Exactly three inner best epochs are required.")
    if any(not 1 <= int(epoch) <= max_epochs for epoch in epochs):
        raise ValueError("An inner best epoch is outside the frozen bounds.")
    value = int(Decimal(str(float(np.median(epochs)))).quantize(Decimal("1"), rounding=ROUND_HALF_UP))
    if not 1 <= value <= max_epochs:
        raise ValueError("Refit epoch is outside the frozen bounds.")
    return value


def build_b0_selection_contract(run_id: str, source_commit: str, inner_checksum: str,
                                feature_contracts: dict[str, Any]) -> dict[str, Any]:
    studies = []
    for track in ("late_stage", "early_warning"):
        for outer_fold in range(5):
            studies.append({"study_id": f"{run_id}:B0:{track}:outer{outer_fold}", "model_family": "B0",
                            "track": track, "outer_fold": outer_fold, "trial_budget": 4,
                            "alphas": [0.01, 0.1, 1.0, 10.0], "expected_inner_evaluations": 12,
                            "objective": "minimize_inner_mean_rmse_raw", "tie_break_1": "minimize_inner_mean_mae_raw",
                            "tie_break_2": "smaller_alpha", "inner_split_manifest_checksum": inner_checksum,
                            "feature_contract_checksum": feature_contracts[track]["semantic_checksum"], "source_commit": source_commit})
    result = {"contract_version": V3_2_PROTOCOL_VERSION, "run_id": run_id, "created_before_compute": True, "studies": studies}
    result["semantic_checksum"] = checksum(result)
    return result


def selection_order(row: pd.Series, objective: str) -> tuple:
    if objective == "maximize_macro_f1":
        return (-float(row.mean_macro_f1), float(row.mean_ordinal_mae), int(row.trial_id))
    return (float(row.mean_rmse_raw), float(row.mean_mae_raw), float(row.alpha))


def validate_selected_trials(studies: list[dict[str, Any]], trials: pd.DataFrame, selected: pd.DataFrame,
                             inner_checksum: str, search_contract: dict[str, Any]) -> dict[str, int]:
    errors = {"missing_selected": 0, "duplicate_selected": 0, "missing_trial": 0, "bad_inner_folds": 0,
              "selected_not_completed": 0, "selected_not_best": 0, "checksum_payload_mismatch": 0,
              "inner_split_mismatch": 0, "search_space_violation": 0}
    expected = {x["study_id"]: x for x in studies}
    if not selected.empty:
        errors["duplicate_selected"] = int(selected.duplicated("study_id", keep=False).sum())
    for study_id, study in expected.items():
        trial_subset = trials[trials.study_id == study_id]
        pick = selected[selected["study_id"] == study_id] if "study_id" in selected.columns else selected
        if pick.empty:
            errors["missing_selected"] += 1; continue
        if study_id not in set(trials.study_id):
            errors["missing_trial"] += 1; continue
        for trial_id in range(study["trial_budget"]):
            entries = trial_subset[trial_subset.trial_id == trial_id]
            if entries.empty:
                errors["missing_trial"] += 1; continue
            if set(entries.inner_fold) != {0, 1, 2}:
                errors["bad_inner_folds"] += 1
        completed = trial_subset[trial_subset.status == "completed"]
        chosen = pick.iloc[0]
        chosen_rows = completed[completed.trial_id == chosen.selected_trial_id]
        if chosen_rows.empty:
            errors["selected_not_completed"] += 1; continue
        payload = json_load(chosen.config_payload)
        if checksum(payload) != chosen.config_checksum:
            errors["checksum_payload_mismatch"] += 1
        if chosen.inner_split_manifest_checksum != inner_checksum:
            errors["inner_split_mismatch"] += 1
        grouped = completed.groupby("trial_id").first().reset_index()
        if study["model_family"] == "B0":
            grouped["mean_rmse_raw"] = completed.groupby("trial_id").rmse_raw.mean().values
            grouped["mean_mae_raw"] = completed.groupby("trial_id").mae_raw.mean().values
            grouped["alpha"] = grouped.apply(lambda x: json_load(x.config_payload)["alpha"], axis=1)
            best = sorted((selection_order(row, "minimize_rmse") for _, row in grouped.iterrows()))[0]
            actual = selection_order(grouped[grouped.trial_id == chosen.selected_trial_id].iloc[0], "minimize_rmse")
        else:
            grouped["mean_macro_f1"] = completed.groupby("trial_id").macro_f1.mean().values
            grouped["mean_ordinal_mae"] = completed.groupby("trial_id").ordinal_mae.mean().values
            best = sorted((selection_order(row, "maximize_macro_f1") for _, row in grouped.iterrows()))[0]
            actual = selection_order(grouped[grouped.trial_id == chosen.selected_trial_id].iloc[0], "maximize_macro_f1")
        if actual != best:
            errors["selected_not_best"] += 1
    return errors


def json_load(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else __import__("json").loads(value)


def validate_authorization(auth: dict[str, Any], expected: dict[str, Any], neural_studies: dict[str, Any],
                           b0_studies: dict[str, Any], inner: dict[str, Any], feature: dict[str, Any],
                           target: dict[str, Any], search: dict[str, Any], acceptance: dict[str, Any],
                           fixed_m4: dict[str, Any], *, source_commit: str, tree_clean: bool) -> None:
    if auth.get("execution_mode") != "full" or auth.get("compute_authorized") is not True:
        raise ValueError("Full execution is not authorized.")
    if not tree_clean or auth.get("source_tree_clean") is not True:
        raise ValueError("Full execution requires a clean tracked source tree.")
    if auth.get("source_commit") != source_commit:
        raise ValueError("Authorization source commit mismatch.")
    contracts = {"expected_job_contract": expected, "selection_study_contract": neural_studies,
                 "b0_selection_contract": b0_studies, "inner_split_manifest": inner,
                 "feature_contract": feature, "target_contract": target, "search_contract": search,
                 "acceptance_contract": acceptance, "fixed_m4_config_contract": fixed_m4}
    for name, contract in contracts.items():
        if contract.get("run_id") not in (None, auth["run_id"]):
            raise ValueError(f"Run ID mismatch in {name}.")
        key = f"{name}_checksum"
        semantic = contract.get("semantic_checksum", checksum(contract))
        if auth.get(key) != semantic:
            raise ValueError(f"Semantic checksum mismatch in {name}.")
    jobs = expected.get("jobs", [])
    if any(job.get("smoke") for job in jobs) or len({j["outer_fold"] for j in jobs}) != 5:
        raise ValueError("Smoke/incomplete-fold expected-job contract cannot run full execution.")
    if {"late_stage", "early_warning"} - {j["track"] for j in jobs}:
        raise ValueError("Both V3 tracks are required.")
    if len(jobs) != 235 or sum(j["expected_record_count"] for j in jobs) != 14852:
        raise ValueError("Expected full job or prediction coverage is incorrect.")


def validate_pooled_oof_exact(predictions: pd.DataFrame, expected_ids: set[str]) -> None:
    for _, group in predictions.dropna(subset=["predicted_g3_raw"]).groupby(["model_family", "track", "training_seed"]):
        if set(group.record_id) != expected_ids or group.record_id.nunique() != len(group):
            raise ValueError("Pooled OOF regression coverage is not exactly the development universe.")
        regression_metric_summary(group.raw_g3.to_numpy(), group.predicted_g3_raw.to_numpy())
