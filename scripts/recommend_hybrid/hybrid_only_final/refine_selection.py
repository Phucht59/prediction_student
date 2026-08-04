"""Refine deterministic config selection from completed inner-validation trials.

No additional search or model fitting is performed.  When at least one config
meets the preregistered 80%/coverage target, coverage remains the first
selection criterion.  When none meets the full target, the fallback selects the
highest inner Precision@1 among configs that still satisfy minimum coverage.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[3]
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import tune_and_evaluate_fast as fast  # noqa: E402

OUT = ROOT / "artifacts/recommend_hybrid/hybrid_only_final"
PROTOCOL = yaml.safe_load(
    (ROOT / "configs/recommend_hybrid/hybrid_only_final_protocol.yaml").read_text(
        encoding="utf-8"
    )
)
CONFIG_KEYS = [
    "risk_weight",
    "evidence_weight",
    "need_weight",
    "certainty_weight",
    "workload_weight",
    "minimum_risk_reduction",
    "maximum_uncertainty",
    "minimum_evidence",
    "minimum_top_margin",
    "minimum_top_score",
]


def _select(trials: pd.DataFrame) -> dict[str, Any]:
    rule = PROTOCOL["selection_rule"]
    target = trials[trials["meets_target"].astype(bool)].copy()
    if len(target):
        target.sort_values(
            [
                "mean_actionable_coverage",
                "worst_stage_precision",
                "mean_macro_action_precision",
                "mean_precision_at_1",
                "mean_selective_accuracy",
                "maximum_top_action_concentration",
                "config_id",
            ],
            ascending=[False, False, False, False, False, True, True],
            kind="stable",
            inplace=True,
        )
        chosen = target.iloc[0]
        reason = "TARGET_MET_MAXIMIZE_COVERAGE"
    else:
        covered = trials[
            trials["mean_actionable_coverage"]
            >= float(rule["minimum_inner_coverage"])
        ].copy()
        pool = covered if len(covered) else trials.copy()
        pool.sort_values(
            [
                "mean_precision_at_1",
                "worst_stage_precision",
                "mean_macro_action_precision",
                "mean_actionable_coverage",
                "mean_selective_accuracy",
                "maximum_top_action_concentration",
                "config_id",
            ],
            ascending=[False, False, False, False, False, True, True],
            kind="stable",
            inplace=True,
        )
        chosen = pool.iloc[0]
        reason = (
            "TARGET_NOT_MET_MAXIMIZE_PRECISION_WITH_COVERAGE"
            if len(covered)
            else "COVERAGE_NOT_MET_MAXIMIZE_PRECISION"
        )
    selected: dict[str, Any] = {key: float(chosen[key]) for key in CONFIG_KEYS}
    selected.update(
        {
            "config_id": str(chosen["config_id"]),
            "inner_target_met": bool(chosen["meets_target"]),
            "selection_reason": reason,
            "inner_precision_at_1": float(chosen["mean_precision_at_1"]),
            "inner_actionable_coverage": float(
                chosen["mean_actionable_coverage"]
            ),
        }
    )
    return selected


def _write(path: Path, payload: object) -> None:
    fast._atomic_json(path, payload)


def main() -> None:
    frame = pd.read_parquet(OUT / "dataset/candidate_rows.parquet")
    selection_out = OUT / "model_selection"
    evaluation_out = OUT / "evaluation"
    oof_rows = []
    fold_metrics = []
    baseline_rows = []

    for outer_fold in PROTOCOL["evaluation"]["outer_folds"]:
        trials = pd.read_csv(selection_out / f"fold_{outer_fold}_trials.csv")
        selected = _select(trials)
        _write(selection_out / f"fold_{outer_fold}_selected.json", selected)
        train = frame[frame["outer_fold"] != outer_fold].copy()
        test = frame[frame["outer_fold"] == outer_fold].copy()
        predictions, metrics, scales = fast._evaluate_partition(
            train, test, selected
        )
        predictions["outer_fold"] = int(outer_fold)
        predictions.to_parquet(
            evaluation_out / f"fold_{outer_fold}_predictions.parquet",
            index=False,
        )
        _write(
            evaluation_out / f"fold_{outer_fold}_metrics.json",
            {"metrics": metrics, "scales": scales, "selected": selected},
        )
        oof_rows.append(predictions)
        fold_metrics.append({"outer_fold": int(outer_fold), **metrics})
        for method, baseline in fast._baseline_metrics(test).items():
            baseline_rows.append(
                {"outer_fold": int(outer_fold), "method": method, **baseline}
            )

    oof = pd.concat(oof_rows, ignore_index=True)
    oof.to_parquet(evaluation_out / "OOF_PREDICTIONS.parquet", index=False)
    issued = oof[oof["issued"] == 1]
    positive_groups = int(oof["group_has_positive"].sum())
    overall = {
        "precision_at_1": float(issued["silver_positive"].mean())
        if len(issued)
        else 0.0,
        "actionable_coverage": float(
            (
                (oof["issued"] == 1)
                & (oof["group_has_positive"] == 1)
            ).sum()
            / positive_groups
        )
        if positive_groups
        else 0.0,
        "issued_groups": int(len(issued)),
        "total_groups": int(len(oof)),
        "positive_groups": positive_groups,
        "action_diversity": int(issued["action_family"].nunique())
        if len(issued)
        else 0,
        "top_action_concentration": float(
            issued["action_family"].value_counts(normalize=True).max()
        )
        if len(issued)
        else 1.0,
    }
    pd.DataFrame(fold_metrics).to_csv(
        evaluation_out / "FOLD_METRICS.csv", index=False
    )
    pd.DataFrame(baseline_rows).to_csv(
        evaluation_out / "BASELINE_METRICS.csv", index=False
    )

    final_trials = pd.read_csv(selection_out / "final_runtime_trials.csv")
    final_selected = _select(final_trials)
    _write(
        OUT / "HYBRID_ONLY_SELECTED_CONFIG.json",
        {
            "status": "SELECTED_NOT_RELEASED",
            "config": final_selected,
            "normalization_scales": fast._fit_scales(frame),
            "additional_learned_model": False,
            "silver_labels_used_at_runtime": False,
            "claim_boundary": PROTOCOL["claim_boundary"],
            "execution_script": "tune_and_evaluate_fast.py",
            "selection_refinement": "refine_selection.py",
        },
    )
    _write(
        evaluation_out / "OOF_RESULTS.json",
        {
            "status": "COMPLETE",
            "overall": overall,
            "folds": fold_metrics,
            "target_precision": PROTOCOL["release_gates"][
                "top1_precision_minimum"
            ],
            "additional_learned_model": False,
            "future_features_in_scoring": False,
            "runtime_filtering_order_matched": True,
            "selection_refined_from_inner_trials_only": True,
        },
    )
    print(json.dumps(overall, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
