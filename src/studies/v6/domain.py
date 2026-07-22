from __future__ import annotations

import json
import re
from typing import Any

import numpy as np
import pandas as pd
import torch
from xgboost import XGBClassifier

from src.studies.v5.common.metrics import binary_metrics_per_record_threshold
from src.studies.v5_1.oulad.data import prepare_oulad_inputs
from src.studies.v5_1.oulad.runner import _load

from .contract import ARTIFACT_ROOT, REPORT_ROOT, ROOT, atomic_json, atomic_text, sha256_file
from .evaluation import _metrics
from .multitask import build_temporal_targets, fit_multitask
from .pretraining import _config, fit_minimal_pretraining


DOMAIN_ROOT = ARTIFACT_ROOT / "prediction/domain_generalization"


def _slug(value: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_-]+", "_", value)


def _xgb_features(inputs) -> np.ndarray:
    return np.concatenate(
        [
            inputs.sequence.reshape(len(inputs.sequence), -1),
            inputs.mask.astype(np.float32),
            inputs.aggregate,
            inputs.static,
        ],
        axis=1,
    )


def evaluate_domain_generalization(device_name: str = "cuda") -> dict[str, Any]:
    output = DOMAIN_ROOT / "run_state.json"
    if output.is_file():
        cached = json.loads(output.read_text(encoding="utf-8"))
        if cached.get("status") == "COMPLETE":
            return cached
    _, _, data = _load()
    targets = build_temporal_targets(data)
    config = _config()
    checkpoint_root = DOMAIN_ROOT / "checkpoints"
    checkpoint_root.mkdir(parents=True, exist_ok=True)
    results_path = DOMAIN_ROOT / "holdout_metrics.json"
    rows: list[dict[str, Any]] = (
        json.loads(results_path.read_text(encoding="utf-8"))
        if results_path.is_file()
        else []
    )
    completed = {(row["protocol"], row["holdout"]) for row in rows}
    cohort = data.base.cohort.reset_index(drop=True)
    modules = sorted(cohort.code_module.astype(str).unique())
    for holdout in modules:
            protocol_name = "leave_one_module_out"
            if (protocol_name, holdout) in completed:
                continue
            test_mask = cohort.code_module.eq(holdout)
            test_index = np.flatnonzero(test_mask.to_numpy())
            train_index = np.flatnonzero(~test_mask.to_numpy())
            if not len(test_index) or len(np.unique(data.y[test_index])) < 2:
                rows.append(
                    {
                        "protocol": protocol_name,
                        "holdout": holdout,
                        "status": "SKIP_SINGLE_CLASS_OR_EMPTY",
                        "records": int(len(test_index)),
                    }
                )
                atomic_json(results_path, rows)
                continue
            train = prepare_oulad_inputs(data, train_index, train_index)
            test = prepare_oulad_inputs(
                data, train_index, test_index, fitted=train.preprocessors
            )
            pretrain = fit_minimal_pretraining(
                train,
                data.dynamic_sequence[train_index],
                dynamic_channel_order=data.dynamic_channel_order,
                config=config,
                seed=42,
                epochs=5,
                device_name=device_name,
            )
            fit = fit_multitask(
                train,
                test,
                targets,
                train_index,
                test_index,
                config=config,
                weights={"survival": 0.15, "outcome": 0.15},
                initial_temporal_state=pretrain.temporal_state_dict,
                seed=42,
                epochs=8,
                device_name=device_name,
            )
            checkpoint = checkpoint_root / f"{_slug(protocol_name)}__{_slug(holdout)}.pt"
            torch.save(fit.state_dict, checkpoint)
            evaluation = pd.DataFrame(
                {
                    "target": data.y[test_index].astype(int),
                    "threshold": np.full(len(test_index), 0.495),
                    "withdrawal_event": targets.withdrawal_event[test_index],
                    "observation_week": targets.observation_week[test_index],
                    "outcome_target": targets.outcome_target[test_index],
                    "withdrawal_day": targets.withdrawal_day[test_index],
                    "cutoff_day": cohort.iloc[test_index].cutoff_day.to_numpy(),
                }
            )
            deep_metrics = _metrics(
                evaluation,
                fit.binary_probability,
                fit.hazard_probability,
                fit.outcome_probability,
            )
            xgb = XGBClassifier(
                n_estimators=200,
                max_depth=4,
                learning_rate=0.05,
                subsample=0.9,
                colsample_bytree=0.9,
                reg_lambda=1.0,
                objective="binary:logistic",
                eval_metric="logloss",
                random_state=42,
                n_jobs=8,
                tree_method="hist",
            )
            xgb.fit(_xgb_features(train), data.y[train_index])
            xgb_probability = xgb.predict_proba(_xgb_features(test))[:, 1]
            xgb_metrics = binary_metrics_per_record_threshold(
                data.y[test_index], xgb_probability, np.full(len(test_index), 0.5)
            )
            common = {
                "status": "COMPLETE",
                "train_records": int(len(train_index)),
                "deep_checkpoint": checkpoint.relative_to(ROOT).as_posix(),
                "deep_checkpoint_sha256": sha256_file(checkpoint),
                "deep_replay_max_abs_difference": fit.replay_max_abs_difference,
                "threshold": 0.495,
                "threshold_source": "median frozen V5.1 inner-OOF thresholds",
                "seed": 42,
                "training_exclusion": "entire_target_module",
            }
            rows.append(
                {
                    "protocol": "leave_one_module_out",
                    "holdout": holdout,
                    "records": int(len(test_index)),
                    "deep": deep_metrics,
                    "xgboost_fixed_cross_check": xgb_metrics,
                    **common,
                }
            )
            test_cohort = cohort.iloc[test_index].reset_index(drop=True)
            for presentation in sorted(test_cohort.code_presentation.astype(str).unique()):
                local = np.flatnonzero(test_cohort.code_presentation.eq(presentation).to_numpy())
                if len(np.unique(data.y[test_index][local])) < 2:
                    continue
                rows.append(
                    {
                        "protocol": "leave_one_presentation_out",
                        "holdout": f"{holdout}::{presentation}",
                        "records": int(len(local)),
                        "deep": _metrics(
                            evaluation.iloc[local].reset_index(drop=True),
                            fit.binary_probability[local],
                            fit.hazard_probability[local],
                            fit.outcome_probability[local],
                        ),
                        "xgboost_fixed_cross_check": binary_metrics_per_record_threshold(
                            data.y[test_index][local],
                            xgb_probability[local],
                            np.full(len(local), 0.5),
                        ),
                        **common,
                    }
                )
            atomic_json(results_path, rows)
    completed_rows = [row for row in rows if row["status"] == "COMPLETE"]
    standard = json.loads(
        (ARTIFACT_ROOT / "prediction/final/run_state.json").read_text(encoding="utf-8")
    )["ensemble_metrics"]
    summary: dict[str, Any] = {}
    for protocol_name in ("leave_one_presentation_out", "leave_one_module_out"):
        selected = [row for row in completed_rows if row["protocol"] == protocol_name]
        summary[protocol_name] = {
            "holdouts": len(selected),
            "deep_macro_f1_mean": float(np.mean([row["deep"]["macro_f1"] for row in selected])),
            "deep_at_risk_f1_mean": float(
                np.mean([row["deep"]["at_risk_f1"] for row in selected])
            ),
            "deep_pr_auc_mean": float(np.mean([row["deep"]["pr_auc"] for row in selected])),
            "deep_brier_mean": float(np.mean([row["deep"]["brier"] for row in selected])),
            "deep_ece_mean": float(np.mean([row["deep"]["ece"] for row in selected])),
            "xgboost_macro_f1_mean": float(
                np.mean([row["xgboost_fixed_cross_check"]["macro_f1"] for row in selected])
            ),
        }
        summary[protocol_name]["macro_f1_drop_vs_standard"] = float(
            summary[protocol_name]["deep_macro_f1_mean"] - standard["macro_f1"]
        )
    deep_mean = float(np.mean([value["deep_macro_f1_mean"] for value in summary.values()]))
    xgb_mean = float(np.mean([value["xgboost_macro_f1_mean"] for value in summary.values()]))
    conclusion = (
        "DEEP_GENERALIZATION_ADVANTAGE"
        if deep_mean >= xgb_mean + 0.003
        else "PRACTICAL_GENERALIZATION_TIE"
        if abs(deep_mean - xgb_mean) < 0.003
        else "NO_GENERALIZATION_ADVANTAGE"
    )
    result = {
        "schema_version": "v6_domain_generalization_v1",
        "status": "COMPLETE",
        "protocol_a_standard": standard,
        "summary": summary,
        "conclusion": conclusion,
        "selection_used_domain_test": False,
        "fixed_seed": 42,
        "presentation_protocol_training": (
            "Each presentation uses the stricter model excluding its entire module; "
            "seven leakage-free module models cover all unseen presentations."
        ),
        "future_accessed": False,
    }
    atomic_json(output, result)
    report_rows = "\n".join(
        f"| {name} | {value['holdouts']} | {value['deep_macro_f1_mean']:.6f} | "
        f"{value['deep_pr_auc_mean']:.6f} | {value['deep_brier_mean']:.6f} | "
        f"{value['xgboost_macro_f1_mean']:.6f} |"
        for name, value in summary.items()
    )
    atomic_text(
        REPORT_ROOT / "DOMAIN_GENERALIZATION_REPORT.md",
        f"""# V6 domain-generalization report

| Protocol | Holdouts | Deep Macro-F1 | Deep PR-AUC | Deep Brier | Fixed XGBoost Macro-F1 |
|---|---:|---:|---:|---:|---:|
{report_rows}

Conclusion: **{conclusion}**. Each holdout uses the frozen Candidate C, seed 42,
five pretraining epochs, eight multi-task epochs, and a threshold frozen before
domain evaluation. Presentation metrics use the stricter corresponding
leave-one-module-out model, so the target presentation and its entire module are
unseen. Holdout records never enter training, threshold, epoch, loss
or architecture selection. This conclusion is separate from the standard
grouped benchmark.
""",
    )
    return result


__all__ = ["evaluate_domain_generalization"]
