from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import yaml

from src.studies import oulad_multistage as study


ROOT = Path(__file__).resolve().parents[2]


def test_preregistered_four_stage_contract() -> None:
    protocol = yaml.safe_load((ROOT / "configs/final/unified_stage_aware_oulad.yaml").read_text())
    assert protocol["stages"].keys() == set(study.STAGES)
    assert protocol["training"]["outer_folds"] == 3
    assert protocol["training"]["inner_folds"] == 2
    assert tuple(protocol["training"]["fixed_seeds"]) == study.SEEDS
    assert protocol["training"]["outer_used_for_tuning"] is False
    assert protocol["training"]["best_seed_selection"] is False


def test_temporal_contract_is_exactly_47_and_future_padding_is_zero() -> None:
    base = np.zeros((2, 3, len(study.BASE_CHANNELS)), dtype=np.float32)
    base[0, 0, 0] = 4.0
    mask = np.asarray([[True, True, False], [True, False, False]])
    values = study._dynamic(base, mask)
    assert values.shape == (2, 3, 47)
    assert np.all(values[~mask] == 0)
    assert tuple(study.CHANNELS[:16]) == study.BASE_CHANNELS


def test_stage_cutoff_order_and_frozen_middle_rule() -> None:
    bundle = study._build_bundle()
    cutoff = study._cutoff_manifest(bundle)
    assert cutoff["monotonicity_pass"].all()
    assert cutoff.loc[cutoff.stage.eq("M1_MIDDLE_FROZEN"), "exact_f2_compatibility"].all()
    for _, group in cutoff.groupby(["code_module", "code_presentation"]):
        values = group.set_index("stage").loc[list(study.STAGES), "cutoff_day"].to_numpy()
        assert np.all(np.diff(values) > 0)


def test_stage_views_exclude_unregistration_from_predictors() -> None:
    bundle = study._build_bundle()
    for view in bundle.stages.values():
        assert "date_unregistration" in view.frame.columns  # eligibility/survival only
        assert "date_unregistration" not in study.STATIC_COLUMNS
        assert view.aggregate.shape[1] == 165


def test_stage_expansion_keeps_base_group_partition() -> None:
    bundle = study._build_bundle()
    base = bundle.base[["base_record_id", "id_student", "outer_fold"]].drop_duplicates()
    assert base.groupby("id_student")["outer_fold"].nunique().eq(1).all()
    for stage in study.STAGES:
        view = bundle.stages[stage].frame
        merged = view.merge(base, on="base_record_id", suffixes=("", "_base"), validate="many_to_one")
        assert (merged.outer_fold == merged.outer_fold_base).all()


def test_final_evidence_validates_when_materialized() -> None:
    path = ROOT / "artifacts" / "final" / "unified_stage_aware_oulad" / "validation.json"
    if not path.is_file():
        return
    evidence = json.loads(path.read_text())
    assert evidence["status"] == "PASS"
    assert evidence["final_training_runs"] == 150
    assert evidence["stage_rows"] == 40


def test_svm_amendment_uses_rbf_without_internal_probability() -> None:
    pipeline = study._make_tabular("svm", 42)
    estimator = pipeline.named_steps["model"]
    assert estimator.kernel == "rbf"
    assert estimator.probability is False
    assert estimator.cache_size == 4096


def test_svm_checkpoint_prediction_rejects_superseded_format(tmp_path) -> None:
    path = tmp_path / "seed_42.joblib"
    joblib.dump({"checkpoint_schema": "old_probability_true"}, path)
    frame = np.zeros((1, 165), dtype=np.float32)
    with np.testing.assert_raises(RuntimeError):
        study._predict_checkpoint(
            "svm",
            path,
            study._build_bundle().stages[study.STAGES[0]].frame.iloc[:1],
            np.zeros((1, 1, 47), dtype=np.float32),
            np.ones(1, dtype=np.int64),
            np.ones((1, 1), dtype=bool),
            frame,
        )
