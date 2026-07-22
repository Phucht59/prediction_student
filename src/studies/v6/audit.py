from __future__ import annotations

import json
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.studies.v5.common.metrics import (
    binary_metrics_per_record_threshold,
    expected_calibration_error,
)
from src.studies.v5_1.oulad.data import OULADInputsV51, prepare_oulad_inputs
from src.studies.v5_1.oulad.models import OULADHybridV51
from src.studies.v5_1.oulad.runner import _inner_splits, _load
from src.studies.v5_1.oulad.training import choose_threshold, fit_prepared_oulad_model

from .contract import ARTIFACT_ROOT, REPORT_ROOT, ROOT, atomic_json, atomic_text, load_protocol


AUDIT_ROOT = ARTIFACT_ROOT / "audit"


def _selected_config() -> dict[str, Any]:
    selected = json.loads(
        (ROOT / "artifacts/v5_1/oulad/selected_configs.json").read_text(encoding="utf-8")
    )[0]
    config = dict(selected["config"])
    config.update(
        {
            "input_projection": 48,
            "conv_channels": 24,
            "kernels": [2, 3],
            "dilation": 2,
            "lstm_hidden": 64,
            "lstm_layers": 1,
            "pooling": "masked_mean_max",
            "pooling_projection": 48,
            "aggregate_hidden": 64,
            "static_hidden": 32,
            "fusion_hidden": 64,
            "fusion": "gated_residual",
            "branch_dropout": 0.0,
            "loss": "standard_bce",
            "batch_size": 256,
            "gradient_clip": 1.0,
            "temporal_order": "original",
        }
    )
    return config


def _destroy_order(inputs: OULADInputsV51, order: str, seed: int) -> OULADInputsV51:
    if order == "original":
        return inputs
    sequence = inputs.sequence.copy()
    rng = np.random.default_rng(seed)
    for row, raw_length in enumerate(inputs.lengths):
        length = int(raw_length)
        valid = sequence[row, :length].copy()
        if order == "reversed":
            replacement = valid[::-1]
        elif order == "shuffled":
            replacement = valid[rng.permutation(length)]
        elif order == "bag_of_weeks":
            replacement = np.repeat(valid.mean(axis=0, keepdims=True), length, axis=0)
        else:
            raise ValueError(f"Unknown order-destruction variant: {order}")
        sequence[row, :length] = replacement
    sequence *= inputs.mask[..., None]
    return replace(inputs, sequence=sequence.astype(np.float32))


def _save_order_progress(predictions: list[dict[str, Any]], metadata: list[dict[str, Any]]) -> None:
    AUDIT_ROOT.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(predictions).to_parquet(AUDIT_ROOT / "order_predictions.parquet", index=False)
    atomic_json(AUDIT_ROOT / "order_checkpoint_metadata.json", metadata)
    atomic_json(
        AUDIT_ROOT / "run_state.json",
        {
            "status": "RUNNING",
            "stage": "order_destruction",
            "completed_fits": len(metadata),
            "outer_test_accessed": False,
            "future_accessed": False,
        },
    )


def _crossfit_metrics(frame: pd.DataFrame) -> dict[str, Any]:
    thresholds = np.zeros(len(frame), dtype=float)
    for fold in sorted(frame.inner_fold.unique()):
        fit = frame.inner_fold.to_numpy() != fold
        score = frame.inner_fold.to_numpy() == fold
        threshold = choose_threshold(
            frame.loc[fit, "target"].to_numpy(), frame.loc[fit, "probability"].to_numpy()
        )["threshold"]
        thresholds[score] = threshold
    return binary_metrics_per_record_threshold(
        frame.target.to_numpy(), frame.probability.to_numpy(), thresholds
    )


def order_destruction_audit(device: str = "cuda") -> dict[str, Any]:
    protocol = load_protocol()
    _, v4_protocol, data = _load()
    splits = _inner_splits(data, 0, v4_protocol)
    config = _selected_config()
    seed = int(protocol["audit"]["order_seed"])
    variants = list(protocol["audit"]["order_variants"])
    prediction_path = AUDIT_ROOT / "order_predictions.parquet"
    metadata_path = AUDIT_ROOT / "order_checkpoint_metadata.json"
    predictions = (
        pd.read_parquet(prediction_path).to_dict(orient="records")
        if prediction_path.is_file()
        else []
    )
    metadata = (
        json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata_path.is_file()
        else []
    )
    completed = {(str(row["variant"]), int(row["inner_fold"])) for row in metadata}
    for variant in variants:
        for inner_fold, (train_index, validation_index) in enumerate(splits):
            if (variant, inner_fold) in completed:
                continue
            train = prepare_oulad_inputs(data, train_index, train_index)
            validation = prepare_oulad_inputs(
                data, train_index, validation_index, fitted=train.preprocessors
            )
            train = _destroy_order(train, variant, seed + inner_fold)
            validation = _destroy_order(validation, variant, seed + inner_fold)
            fit = fit_prepared_oulad_model(
                train,
                validation,
                config=config,
                seed=seed,
                fixed_epochs=8,
                device_name=device,
            )
            metadata.append(
                {
                    "variant": variant,
                    "outer_training_fold": 0,
                    "inner_fold": inner_fold,
                    "seed": seed,
                    "fixed_epochs": 8,
                    "selected_epoch": fit.selected_epoch,
                    "parameter_count": fit.parameter_count,
                    "runtime_seconds": fit.runtime_seconds,
                    "gpu_peak_memory_bytes": fit.gpu_peak_memory_bytes,
                    "state_dict_sha256": fit.checkpoint_sha256,
                    "replay_max_abs_difference": fit.replay_max_abs_difference,
                }
            )
            predictions.extend(
                {
                    "variant": variant,
                    "inner_fold": inner_fold,
                    "record_id": str(data.base.record_ids[index]),
                    "id_student": int(data.groups[index]),
                    "target": int(data.y[index]),
                    "probability": float(probability),
                }
                for index, probability in zip(validation_index, fit.probability)
            )
            _save_order_progress(predictions, metadata)
    frame = pd.DataFrame(predictions)
    rows = {variant: _crossfit_metrics(group.reset_index(drop=True)) for variant, group in frame.groupby("variant")}
    original = rows["original"]
    for variant, metrics in rows.items():
        metrics["delta_macro_f1_vs_original"] = float(metrics["macro_f1"] - original["macro_f1"])
        metrics["delta_at_risk_f1_vs_original"] = float(
            metrics["at_risk_f1"] - original["at_risk_f1"]
        )
        metrics["delta_pr_auc_vs_original"] = float(metrics["pr_auc"] - original["pr_auc"])
        metrics["delta_brier_vs_original"] = float(metrics["brier"] - original["brier"])
    best_destroyed = max(
        float(rows[name]["macro_f1"]) for name in variants if name != "original"
    )
    delta = float(original["macro_f1"] - best_destroyed)
    verdict = (
        "TEMPORAL_ORDER_HIGH_VALUE"
        if delta >= 0.005
        else "TEMPORAL_ORDER_MODERATE_VALUE"
        if delta >= 0.001
        else "TEMPORAL_ORDER_LOW_VALUE"
    )
    result = {
        "status": "PASS",
        "scope": "outer_training_fold_0_inner_oof",
        "seed": seed,
        "fixed_epochs": 8,
        "threshold_protocol": "cross_fitted_inner_oof",
        "same_configuration": True,
        "same_training_budget": True,
        "variants": rows,
        "original_minus_best_destroyed_macro_f1": delta,
        "verdict": verdict,
        "outer_test_accessed": False,
        "future_accessed": False,
    }
    atomic_json(AUDIT_ROOT / "order_destruction.json", result)
    return result


def _temporal_embeddings() -> pd.DataFrame:
    output = AUDIT_ROOT / "v5_1_temporal_embeddings.parquet"
    if output.is_file():
        return pd.read_parquet(output)
    _, v4_protocol, data = _load()
    splits = _inner_splits(data, 0, v4_protocol)
    config = _selected_config()
    metadata = json.loads(
        (ROOT / "artifacts/v5_2/oulad/screening_checkpoint_metadata.json").read_text(
            encoding="utf-8"
        )
    )
    rows: list[dict[str, Any]] = []
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for inner_fold, (train_index, validation_index) in enumerate(splits):
        validation = prepare_oulad_inputs(data, train_index, validation_index)
        item = next(
            row
            for row in metadata
            if row["candidate"] == "v5_1_serial_baseline"
            and int(row["inner_fold"]) == inner_fold
            and int(row["seed"]) == 42
        )
        model = OULADHybridV51(
            validation.sequence.shape[2],
            validation.aggregate.shape[1],
            validation.static.shape[1],
            config,
        ).to(device)
        state = torch.load(ROOT / str(item["path"]), map_location=device, weights_only=True)
        model.load_state_dict(state)
        model.eval()
        embeddings: list[np.ndarray] = []
        with torch.no_grad():
            for start in range(0, len(validation.target), 512):
                stop = min(start + 512, len(validation.target))
                sequence = torch.from_numpy(validation.sequence[start:stop]).to(device)
                lengths = torch.from_numpy(validation.lengths[start:stop]).to(device)
                mask = torch.from_numpy(validation.mask[start:stop]).to(device)
                temporal, _, _ = model.temporal(sequence, lengths, mask)
                embeddings.append(model.temporal_projection(temporal).cpu().numpy())
        values = np.concatenate(embeddings)
        for local, index in enumerate(validation_index):
            row: dict[str, Any] = {
                "record_id": str(data.base.record_ids[index]),
                "id_student": int(data.groups[index]),
                "code_module": str(data.base.cohort.iloc[index].code_module),
                "inner_fold": inner_fold,
            }
            row.update({f"embedding_{column}": float(value) for column, value in enumerate(values[local])})
            rows.append(row)
    frame = pd.DataFrame(rows)
    frame.to_parquet(output, index=False)
    return frame


def residual_and_oracle_audit() -> tuple[dict[str, Any], dict[str, Any]]:
    embedding = _temporal_embeddings()
    teacher = pd.read_parquet(ROOT / "artifacts/v5_4/oulad/teacher_oof_predictions.parquet")
    deep = pd.read_parquet(ROOT / "artifacts/v5_2/oulad/screening_predictions.parquet")
    deep = deep[(deep.candidate == "v5_1_serial_baseline") & (deep.seed == 42)].copy()
    merged = embedding.merge(
        teacher[
            [
                "record_id",
                "target_at_risk",
                "teacher_probability",
                "teacher_predicted_label",
                "teacher_correctness_training_oof",
            ]
        ],
        on="record_id",
        validate="one_to_one",
    ).merge(
        deep[["record_id", "target", "probability"]].rename(
            columns={"probability": "deep_probability"}
        ),
        on="record_id",
        validate="one_to_one",
    )
    if not np.array_equal(merged.target.to_numpy(), merged.target_at_risk.to_numpy()):
        raise RuntimeError("Residual audit target mismatch")
    features = merged.filter(like="embedding_").to_numpy()
    residual_target = merged.teacher_correctness_training_oof.to_numpy(dtype=int)
    residual_probability = np.zeros(len(merged), dtype=float)
    for fold in sorted(merged.inner_fold.unique()):
        train = merged.inner_fold.to_numpy() != fold
        validation = ~train
        classifier = make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=2000, class_weight="balanced", random_state=3407),
        )
        classifier.fit(features[train], residual_target[train])
        residual_probability[validation] = classifier.predict_proba(features[validation])[:, 1]
    module_segments = {}
    for module, group in merged.assign(residual_probability=residual_probability).groupby("code_module"):
        y = group.teacher_correctness_training_oof.to_numpy(dtype=int)
        p = group.residual_probability.to_numpy()
        module_segments[str(module)] = {
            "records": int(len(group)),
            "teacher_correct_rate": float(y.mean()),
            "residual_auc": float(roc_auc_score(y, p)) if len(np.unique(y)) == 2 else None,
        }
    residual_auc = float(roc_auc_score(residual_target, residual_probability))
    residual_result = {
        "status": "PASS",
        "scope": "outer_training_fold_0_cross_fitted_inner_oof",
        "classifier_input": "v5_1_temporal_projection_embedding_64",
        "classifier": "standardized_logistic_regression",
        "records": int(len(merged)),
        "residual_error_auc": residual_auc,
        "residual_error_pr_auc": float(
            average_precision_score(residual_target, residual_probability)
        ),
        "brier": float(brier_score_loss(residual_target, residual_probability)),
        "ece": expected_calibration_error(residual_target, residual_probability),
        "module_segments": module_segments,
        "verdict": (
            "RESIDUAL_SIGNAL_HIGH"
            if residual_auc >= 0.65
            else "RESIDUAL_SIGNAL_MODERATE"
            if residual_auc >= 0.57
            else "RESIDUAL_SIGNAL_LOW"
        ),
        "complex_selector_allowed": residual_auc >= 0.57,
        "outer_test_accessed": False,
        "future_accessed": False,
    }
    y = merged.target.to_numpy(dtype=int)
    deep_correct = (merged.deep_probability.to_numpy() >= 0.5).astype(int) == y
    xgb_correct = merged.teacher_predicted_label.to_numpy(dtype=int) == y
    oracle_result = {
        "status": "PASS",
        "scope": "outer_training_fold_0_cross_fitted_inner_oof",
        "records": int(len(merged)),
        "deep_correct_xgboost_wrong": int((deep_correct & ~xgb_correct).sum()),
        "xgboost_correct_deep_wrong": int((xgb_correct & ~deep_correct).sum()),
        "both_correct": int((deep_correct & xgb_correct).sum()),
        "both_wrong": int((~deep_correct & ~xgb_correct).sum()),
        "disagreement_rate": float(
            ((merged.deep_probability.to_numpy() >= 0.5) != merged.teacher_predicted_label).mean()
        ),
        "deep_accuracy": float(deep_correct.mean()),
        "xgboost_accuracy": float(xgb_correct.mean()),
        "oracle_union_accuracy": float((deep_correct | xgb_correct).mean()),
        "oracle_gain_over_best": float(
            (deep_correct | xgb_correct).mean() - max(deep_correct.mean(), xgb_correct.mean())
        ),
        "selector_allowed": bool(
            (deep_correct | xgb_correct).mean() - max(deep_correct.mean(), xgb_correct.mean())
            >= 0.005
        ),
        "outer_test_accessed": False,
        "future_accessed": False,
    }
    atomic_json(AUDIT_ROOT / "residual_ceiling.json", residual_result)
    atomic_json(AUDIT_ROOT / "oracle_complementarity.json", oracle_result)
    return residual_result, oracle_result


def survival_data_audit() -> dict[str, Any]:
    _, _, data = _load()
    cohort = data.development_manifest.merge(
        data.base.cohort[
            ["record_id", "cutoff_day", "module_presentation_length"]
        ],
        on="record_id",
        validate="one_to_one",
    )
    registration = pd.read_csv(ROOT / "data/raw/studentRegistration.csv")
    info = pd.read_csv(ROOT / "data/raw/studentInfo.csv")
    keys = ["code_module", "code_presentation", "id_student"]
    frame = cohort.merge(
        registration[keys + ["date_unregistration"]], on=keys, validate="one_to_one"
    ).merge(info[keys + ["final_result"]], on=keys, validate="one_to_one")
    withdrawal = frame.final_result.eq("Withdrawn")
    timestamp = frame.date_unregistration.notna()
    within_course = timestamp & frame.date_unregistration.between(
        0, frame.module_presentation_length, inclusive="both"
    )
    after_cutoff = within_course & (frame.date_unregistration > frame.cutoff_day)
    valid_event = withdrawal & within_course
    result = {
        "status": "PASS" if int(valid_event.sum()) > 0 else "FAIL",
        "records": int(len(frame)),
        "withdrawn_records": int(withdrawal.sum()),
        "withdrawn_with_timestamp": int((withdrawal & timestamp).sum()),
        "withdrawn_with_valid_course_timestamp": int(valid_event.sum()),
        "withdrawal_events_after_f2_cutoff": int((valid_event & after_cutoff).sum()),
        "censored_at_cutoff_or_observation_end": int((~valid_event | ~after_cutoff).sum()),
        "nonwithdrawal_with_unregistration_timestamp": int((~withdrawal & timestamp).sum()),
        "fail_as_withdrawal_event": False,
        "fail_event_time_available": False,
        "weekly_risk_sets_feasible": bool(int((valid_event & after_cutoff).sum()) >= 100),
        "verdict": (
            "WITHDRAWAL_SURVIVAL_FEASIBLE"
            if int((valid_event & after_cutoff).sum()) >= 100
            else "WITHDRAWAL_SURVIVAL_NOT_FEASIBLE"
        ),
        "outer_test_labels_used_for_selection": False,
        "future_accessed": False,
    }
    atomic_json(AUDIT_ROOT / "survival_feasibility.json", result)
    return result


def graph_context_audit() -> dict[str, Any]:
    assessments = pd.read_csv(ROOT / "data/raw/assessments.csv")
    resources = pd.read_csv(ROOT / "data/raw/vle.csv")
    keys = ["code_module", "code_presentation"]
    assessment_summary = assessments.groupby(keys).agg(
        assessment_count=("id_assessment", "nunique"),
        assessment_type_count=("assessment_type", "nunique"),
        assessment_total_weight=("weight", "sum"),
    )
    resource_summary = resources.groupby(keys).agg(
        resource_count=("id_site", "nunique"),
        activity_type_count=("activity_type", "nunique"),
    )
    summary = assessment_summary.join(resource_summary, how="outer").fillna(0).reset_index()
    descriptors = [
        "assessment_count",
        "assessment_type_count",
        "assessment_total_weight",
        "resource_count",
        "activity_type_count",
    ]
    within_module_variation = {
        name: int(summary.groupby("code_module")[name].nunique().gt(1).sum())
        for name in descriptors
    }
    existing_static = {
        "code_module",
        "presentation_season",
        "module_presentation_length",
    }
    existing_sequence = {
        "unique_sites",
        "unique_activity_types",
        "assessment_related_clicks",
    }
    graph_only = {
        "assessment_count",
        "assessment_type_count",
        "assessment_total_weight",
        "resource_count",
        "activity_type_count",
    } - existing_static - existing_sequence
    novel_varying = [name for name in graph_only if within_module_variation[name] > 0]
    passed = len(novel_varying) >= 2
    result = {
        "status": "PASS",
        "presentations": int(len(summary)),
        "modules": int(summary.code_module.nunique()),
        "assessment_nodes": int(assessments.id_assessment.nunique()),
        "resource_nodes": int(resources.id_site.nunique()),
        "assessment_types": int(assessments.assessment_type.nunique()),
        "activity_types": int(resources.activity_type.nunique()),
        "descriptor_columns": descriptors,
        "graph_only_descriptors": sorted(graph_only),
        "within_module_varying_presentations": within_module_variation,
        "novel_varying_descriptors": sorted(novel_varying),
        "verdict": "GRAPH_CONTEXT_PASS" if passed else "GRAPH_CONTEXT_FAIL_REDUNDANT",
        "lightweight_graph_allowed": passed,
        "student_test_nodes_in_graph": False,
        "future_accessed": False,
    }
    atomic_json(AUDIT_ROOT / "graph_context.json", result)
    return result


def _report(result: dict[str, Any]) -> str:
    order = result["order_destruction"]
    residual = result["residual_ceiling"]
    oracle = result["oracle_complementarity"]
    survival = result["survival"]
    graph = result["graph"]
    rows = []
    for name, metrics in order["variants"].items():
        rows.append(
            f"| {name} | {metrics['macro_f1']:.6f} | {metrics['at_risk_f1']:.6f} | "
            f"{metrics['pr_auc']:.6f} | {metrics['brier']:.6f} | "
            f"{metrics['delta_macro_f1_vs_original']:+.6f} |"
        )
    return f"""# V6 knowledge audit

All selection-facing analyses below use only outer-training fold 0 and its
cross-fitted inner folds. Future OULAD remained locked and no outer-test result
was used for a gate.

## Temporal order destruction

| Variant | Macro-F1 | At-risk F1 | PR-AUC | Brier | Δ Macro-F1 |
|---|---:|---:|---:|---:|---:|
{chr(10).join(rows)}

Verdict: **{order['verdict']}**. Every variant used the same V5.1 architecture,
seed, eight-epoch budget and cross-fitted threshold protocol.

## Residual ceiling

The diagnostic classifier used the 64-dimensional cross-fitted V5.1 temporal
projection to predict whether XGBoost was correct. Residual AUC was
`{residual['residual_error_auc']:.6f}` and residual PR-AUC was
`{residual['residual_error_pr_auc']:.6f}`. Verdict:
**{residual['verdict']}**. Complex selector allowed:
`{str(residual['complex_selector_allowed']).lower()}`.

## Oracle complementarity

- Deep correct / XGBoost wrong: {oracle['deep_correct_xgboost_wrong']}
- XGBoost correct / Deep wrong: {oracle['xgboost_correct_deep_wrong']}
- Both correct: {oracle['both_correct']}
- Both wrong: {oracle['both_wrong']}
- Disagreement rate: {oracle['disagreement_rate']:.6f}
- Oracle-union accuracy: {oracle['oracle_union_accuracy']:.6f}
- Oracle gain over best: {oracle['oracle_gain_over_best']:+.6f}

The oracle is diagnostic only and is not a deployable selector.

## Survival feasibility

Valid withdrawal timestamps exist for {survival['withdrawn_with_valid_course_timestamp']}
historical records; {survival['withdrawal_events_after_f2_cutoff']} events occur
after the F2 cutoff. Fail is not treated as a withdrawal event. Verdict:
**{survival['verdict']}**.

## Graph context

The audit found {len(graph['novel_varying_descriptors'])} graph-only descriptors
with within-module presentation variation. Verdict: **{graph['verdict']}**.
This gate only permits a small context embedding; it does not select one.
"""


def run_knowledge_audit(device: str = "cuda") -> dict[str, Any]:
    started = time.perf_counter()
    order = order_destruction_audit(device)
    residual, oracle = residual_and_oracle_audit()
    survival = survival_data_audit()
    graph = graph_context_audit()
    result = {
        "schema_version": "v6_knowledge_audit_v1",
        "status": "PASS" if all(item["status"] == "PASS" for item in [order, residual, oracle, survival, graph]) else "FAIL",
        "order_destruction": order,
        "residual_ceiling": residual,
        "oracle_complementarity": oracle,
        "survival": survival,
        "graph": graph,
        "runtime_seconds": time.perf_counter() - started,
        "outer_test_accessed": False,
        "future_accessed": False,
    }
    atomic_json(AUDIT_ROOT / "knowledge_audit.json", result)
    atomic_text(REPORT_ROOT / "KNOWLEDGE_AUDIT.md", _report(result))
    atomic_json(
        AUDIT_ROOT / "run_state.json",
        {
            "status": result["status"],
            "stage": "knowledge_audit",
            "outer_test_accessed": False,
            "future_accessed": False,
        },
    )
    return result


__all__ = [
    "graph_context_audit",
    "order_destruction_audit",
    "residual_and_oracle_audit",
    "run_knowledge_audit",
    "survival_data_audit",
]

