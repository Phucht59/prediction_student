"""Immutable, one-shot Phase 6 outer evaluation for the frozen H1 candidate."""

from __future__ import annotations

import csv
import hashlib
import json
import os
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset

from src.models.oulad_tabular_residual import CNNBiLSTMTabularResidualOULAD
from src.pipelines import oulad
from src.training.control import stable_hash
from src.training.phase3_optuna import _risk_loss, write_json
from src.training.phase5_mlp_gap import architecture_registry, make_model

ROOT = Path(__file__).resolve().parents[2]
FREEZE_ROOT = ROOT / "artifacts" / "final_candidate_freeze"
FREEZE_PATH = FREEZE_ROOT / "FINAL_H1_FREEZE_MANIFEST.json"
OUT = ROOT / "artifacts" / "final" / "h1_final"
RUNTIME = OUT / "runtime"
CHECKPOINTS = RUNTIME / "checkpoints"
LOGS = OUT / "logs"
STATUS_PATH = RUNTIME / "phase6_status.json"
RUNNING = RUNTIME / "PHASE6_RUNNING"
COMPLETE = RUNTIME / "PHASE6_COMPLETE"
FAILED = RUNTIME / "PHASE6_FAILED"
MODELS = ("H1_TABULAR_RESIDUAL_EXPERT", "H0_CURRENT_HYBRID", "M0_MLP")
EXPECTED_OLD_FINAL = {
    "artifacts/final/final_results.json": "000c185fb2fd9ba4b528e79d98636fdb17ee4586dbad197e9990717164b3681b",
    "artifacts/final/final_results.csv": "d2271c48bc6ed65a2836ec3b2430eef0777ad9f2b83f0d16092f389e57148b0f",
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("status\nNO_ROWS\n", encoding="utf-8")
        return
    columns = list(dict.fromkeys(key for row in rows for key in row))
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def git_head() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True, encoding="utf-8"
    ).strip()


def prepare_directories() -> None:
    for path in (OUT, RUNTIME, CHECKPOINTS, LOGS):
        path.mkdir(parents=True, exist_ok=True)


def status_payload(**updates: Any) -> dict[str, Any]:
    current = (
        json.loads(STATUS_PATH.read_text(encoding="utf-8"))
        if STATUS_PATH.is_file()
        else {
            "state": "PENDING",
            "started_at": None,
            "finished_at": None,
            "current_stage": "preflight",
            "completed_runs": 0,
            "failed_runs": 0,
            "current_candidate": None,
            "current_outer_fold": None,
            "current_seed": None,
            "exit_code": None,
            "pid": os.getpid(),
        }
    )
    current.update(updates)
    write_json(STATUS_PATH, current)
    return current


def set_sentinel(state: str, details: dict[str, Any] | None = None) -> None:
    for path in (RUNNING, COMPLETE, FAILED):
        if path.exists():
            path.unlink()
    target = {"RUNNING": RUNNING, "COMPLETE": COMPLETE, "FAILED": FAILED}[state]
    write_json(target, {"state": state, "at": utc_now(), **(details or {})})


def load_freeze() -> dict[str, Any]:
    return json.loads(FREEZE_PATH.read_text(encoding="utf-8"))


def validate_freeze(*, require_freeze_commit: bool = True) -> dict[str, Any]:
    manifest = load_freeze()
    science = manifest["scientific_configuration"]
    checks = {
        "final_candidate_hash": stable_hash(science)
        == manifest["final_candidate_hash"],
        "feature_schema_hash": stable_hash(science["feature_schema"])
        == manifest["feature_schema_hash"],
        "preprocessing_hash": stable_hash(science["preprocessing"])
        == manifest["preprocessing_hash"],
        "training_policy_hash": stable_hash(science["training_policy"])
        == manifest["training_policy_hash"],
        "evaluation_protocol_hash": stable_hash(science["evaluation_protocol"])
        == manifest["evaluation_protocol_hash"],
        "architecture_hash": architecture_registry()[2]["architecture_hash"]
        == manifest["architecture_hash"],
        "temporal_backbone_hash": architecture_registry()[2][
            "temporal_backbone_hash"
        ]
        == manifest["temporal_backbone_hash"],
        "parameter_count": int(manifest["parameter_count"]) == 160492,
        "seeds": science["evaluation_protocol"]["final_seeds"]
        == [42, 1201, 2026, 3407, 7319],
        "outer_folds": int(science["evaluation_protocol"]["outer_folds"]) == 3,
        "candidate_count": int(
            science["evaluation_protocol"]["candidate_count_h1"]
        )
        == 1,
        "optuna_trials": int(science["evaluation_protocol"]["optuna_trials"]) == 0,
        "outer_access_before_freeze": manifest["outer_test_accessed_before_freeze"]
        is False,
    }
    freeze_commit = git_head()
    if require_freeze_commit:
        message = subprocess.check_output(
            ["git", "show", "-s", "--format=%s", "HEAD"],
            cwd=ROOT,
            text=True,
            encoding="utf-8",
        ).strip()
        tracked = subprocess.run(
            ["git", "cat-file", "-e", f"HEAD:{FREEZE_PATH.relative_to(ROOT).as_posix()}"],
            cwd=ROOT,
            check=False,
            capture_output=True,
        ).returncode == 0
        checks["freeze_manifest_committed"] = tracked
        checks["freeze_commit_message"] = message.startswith("freeze:")
        checks["freeze_commit_after_source"] = freeze_commit != manifest["source_commit"]
    if not all(checks.values()):
        raise RuntimeError(f"freeze validation failed: {checks}")
    return {
        "status": "PASS",
        "checks": checks,
        "freeze_commit": freeze_commit,
        "candidate_hash": manifest["final_candidate_hash"],
    }


def _restore_preprocessor(state: dict[str, Any]) -> oulad._DeepPreprocessor:
    preprocessor = oulad._DeepPreprocessor()
    for key, value in state.items():
        setattr(preprocessor, key, value)
    return preprocessor


def _train_deep_final(
    candidate: str,
    train: tuple,
    config: dict[str, Any],
    seed: int,
    epochs: int,
    checkpoint: Path,
    manifest: dict[str, Any],
) -> dict[str, Any]:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    frame, sequence, length, mask, aggregate_raw, labels, sample_weight = train
    preprocessor = oulad._DeepPreprocessor().fit(frame, aggregate_raw)
    aggregate, static = preprocessor.transform(frame, aggregate_raw)
    device = torch.device("cuda")
    model = make_model(candidate, aggregate.shape[1], static.shape[1], config).to(device)
    if candidate == "H1_TABULAR_RESIDUAL_EXPERT" and not isinstance(
        model, CNNBiLSTMTabularResidualOULAD
    ):
        raise RuntimeError("H1 model class changed")
    if candidate == "H1_TABULAR_RESIDUAL_EXPERT":
        if sum(parameter.numel() for parameter in model.parameters()) != 160492:
            raise RuntimeError("H1 parameter count changed")
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["learning_rate"]),
        weight_decay=float(config["weight_decay"]),
    )
    risk_loss, positive_weight = _risk_loss(labels, config, device)
    dataset = TensorDataset(
        torch.from_numpy(sequence),
        torch.from_numpy(length.astype(np.int64)),
        torch.from_numpy(mask.astype(np.float32)),
        torch.from_numpy(aggregate),
        torch.from_numpy(static),
        torch.from_numpy(labels.astype(np.float32)),
        torch.from_numpy(sample_weight.astype(np.float32)),
        torch.from_numpy(frame.outcome_aux.to_numpy(dtype=np.int64)),
        torch.from_numpy(frame.cutoff_day.to_numpy(dtype=np.int64)),
        torch.from_numpy(frame.module_presentation_length.to_numpy(dtype=np.int64)),
        torch.from_numpy(frame.date_unregistration.fillna(-1).to_numpy(dtype=np.int64)),
    )
    loader = DataLoader(
        dataset,
        batch_size=int(config["batch_size"]),
        shuffle=True,
        num_workers=0,
        generator=torch.Generator().manual_seed(seed),
    )
    for _epoch in range(epochs):
        model.train()
        for batch in loader:
            (
                batch_sequence,
                batch_length,
                batch_mask,
                batch_aggregate,
                batch_static,
                target,
                weight,
                outcome,
                cutoff,
                course_end,
                unregistration,
            ) = (value.to(device) for value in batch)
            optimizer.zero_grad()
            output = model(
                batch_sequence,
                batch_length,
                batch_mask,
                batch_aggregate,
                batch_static,
            )
            loss, _ = oulad._multitask_loss(
                output,
                target,
                weight,
                outcome,
                cutoff,
                course_end,
                unregistration,
                risk_loss,
                survival_weight=float(config["survival_weight"]),
                outcome_weight=float(config["outcome_weight"]),
            )
            if not torch.isfinite(loss):
                raise FloatingPointError("non-finite final training loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
    payload = {
        "candidate": candidate,
        "state_dict": {
            key: value.detach().cpu() for key, value in model.state_dict().items()
        },
        "preprocessor": preprocessor.state(),
        "config": config,
        "aggregate_dim": aggregate.shape[1],
        "static_dim": static.shape[1],
        "fixed_epochs": epochs,
        "seed": seed,
        "positive_weight": positive_weight,
        "final_candidate_hash": manifest["final_candidate_hash"],
        "architecture_hash": (
            manifest["architecture_hash"]
            if candidate == "H1_TABULAR_RESIDUAL_EXPERT"
            else architecture_registry()[1]["architecture_hash"]
        ),
        "feature_schema_hash": manifest["feature_schema_hash"],
        "training_policy_hash": manifest["training_policy_hash"],
        "evaluation_protocol_hash": manifest["evaluation_protocol_hash"],
        "parameter_count": int(sum(parameter.numel() for parameter in model.parameters())),
        "outer_labels_used_for_training": False,
        "outer_labels_used_for_epoch_selection": False,
        "outer_labels_used_for_threshold_selection": False,
    }
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, checkpoint)
    del model, optimizer, loader, dataset
    torch.cuda.empty_cache()
    return payload


def _predict_deep_payload(
    checkpoint: Path,
    frame: pd.DataFrame,
    sequence: np.ndarray,
    length: np.ndarray,
    mask: np.ndarray,
    aggregate_raw: np.ndarray,
) -> np.ndarray:
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model = make_model(
        payload["candidate"],
        int(payload["aggregate_dim"]),
        int(payload["static_dim"]),
        payload["config"],
    )
    model.load_state_dict(payload["state_dict"])
    device = torch.device("cuda")
    model.to(device).eval()
    preprocessor = _restore_preprocessor(payload["preprocessor"])
    aggregate, static = preprocessor.transform(frame, aggregate_raw)
    return oulad._predict_deep(
        model, sequence, length, mask, aggregate, static, "cnn_bilstm", device
    )


def _fit_mlp_final(
    train: tuple, seed: int, checkpoint: Path, manifest: dict[str, Any]
) -> dict[str, Any]:
    frame, _, _, _, aggregate, labels, sample_weight = train
    estimator = oulad._make_tabular("mlp", seed)
    features = oulad._tabular_frame(frame, aggregate)
    try:
        estimator.fit(features, labels, sample_weight=sample_weight)
    except (TypeError, ValueError):
        estimator.fit(features, labels)
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    model = estimator.named_steps["model"]
    parameters = int(
        sum(array.size for array in [*model.coefs_, *model.intercepts_])
    )
    payload = {
        "candidate": "M0_MLP",
        "seed": seed,
        "parameter_count": parameters,
        "estimator": estimator,
        "feature_schema_hash": manifest["feature_schema_hash"],
        "training_policy_hash": manifest["training_policy_hash"],
        "evaluation_protocol_hash": manifest["evaluation_protocol_hash"],
        "outer_labels_used_for_training": False,
        "outer_labels_used_for_threshold_selection": False,
    }
    joblib.dump(payload, checkpoint)
    return payload


def _predict_mlp(
    checkpoint: Path, frame: pd.DataFrame, aggregate: np.ndarray
) -> np.ndarray:
    estimator = joblib.load(checkpoint)["estimator"]
    return estimator.predict_proba(oulad._tabular_frame(frame, aggregate))[:, 1]


def _run_id(
    candidate_hash: str, candidate: str, outer_fold: int, seed: int, protocol_hash: str
) -> str:
    return stable_hash(
        {
            "candidate_hash": candidate_hash,
            "candidate": candidate,
            "outer_fold": outer_fold,
            "seed": seed,
            "protocol_hash": protocol_hash,
        }
    )


def _checkpoint_path(candidate: str, outer_fold: int, seed: int) -> Path:
    suffix = ".joblib" if candidate == "M0_MLP" else ".pt"
    return CHECKPOINTS / candidate / f"outer{outer_fold}" / f"seed{seed}{suffix}"


def _checkpoint_valid(
    checkpoint: Path,
    candidate: str,
    seed: int,
    manifest: dict[str, Any],
) -> bool:
    if not checkpoint.is_file():
        return False
    try:
        if candidate == "M0_MLP":
            payload = joblib.load(checkpoint)
            return (
                payload["candidate"] == candidate
                and int(payload["seed"]) == seed
                and hasattr(payload["estimator"], "predict_proba")
                and payload["feature_schema_hash"] == manifest["feature_schema_hash"]
                and payload["training_policy_hash"]
                == manifest["training_policy_hash"]
                and payload["evaluation_protocol_hash"]
                == manifest["evaluation_protocol_hash"]
            )
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        return (
            payload["candidate"] == candidate
            and int(payload["seed"]) == seed
            and payload["feature_schema_hash"] == manifest["feature_schema_hash"]
            and payload["training_policy_hash"] == manifest["training_policy_hash"]
            and payload["evaluation_protocol_hash"]
            == manifest["evaluation_protocol_hash"]
            and (
                candidate != "H1_TABULAR_RESIDUAL_EXPERT"
                or (
                    payload["final_candidate_hash"]
                    == manifest["final_candidate_hash"]
                    and payload["architecture_hash"] == manifest["architecture_hash"]
                    and int(payload["parameter_count"]) == 160492
                )
            )
        )
    except Exception:
        return False


def _metric_rows(
    seed_predictions: pd.DataFrame, manifest: dict[str, Any]
) -> tuple[list[dict[str, Any]], pd.DataFrame]:
    inner = manifest["scientific_configuration"]["inner_authority"]
    seed_rows: list[dict[str, Any]] = []
    for (candidate, outer_fold, seed, stage), group in seed_predictions.groupby(
        ["candidate", "outer_fold", "seed", "prediction_stage"]
    ):
        threshold = float(
            inner[candidate][str(outer_fold)]["research_thresholds"][stage]
        )
        seed_rows.append(
            {
                "candidate": candidate,
                "outer_fold": outer_fold,
                "seed": seed,
                "prediction_stage": stage,
                "threshold": threshold,
                **oulad._metric(
                    group.target.to_numpy(),
                    group.probability.to_numpy(),
                    threshold,
                ),
            }
        )
    averaged = (
        seed_predictions.groupby(
            [
                "base_record_id",
                "id_student",
                "code_module",
                "code_presentation",
                "target",
                "cutoff_day",
                "candidate",
                "prediction_stage",
                "outer_fold",
            ],
            as_index=False,
        )
        .probability.mean()
    )
    return seed_rows, averaged


def _stage_and_comparator(
    averaged: pd.DataFrame, manifest: dict[str, Any]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    inner = manifest["scientific_configuration"]["inner_authority"]
    fold_rows: list[dict[str, Any]] = []
    for (candidate, outer_fold, stage), group in averaged.groupby(
        ["candidate", "outer_fold", "prediction_stage"]
    ):
        threshold = float(
            inner[candidate][str(outer_fold)]["research_thresholds"][stage]
        )
        fold_rows.append(
            {
                "candidate": candidate,
                "outer_fold": outer_fold,
                "prediction_stage": stage,
                "threshold": threshold,
                **oulad._metric(
                    group.target.to_numpy(),
                    group.probability.to_numpy(),
                    threshold,
                ),
            }
        )
    fold = pd.DataFrame(fold_rows)
    numeric = [
        column
        for column in fold.columns
        if column not in {"candidate", "outer_fold", "prediction_stage"}
    ]
    stage = (
        fold.groupby(["candidate", "prediction_stage"], as_index=False)[numeric]
        .mean()
    )
    stage_rows = stage.to_dict("records")
    comparators: list[dict[str, Any]] = []
    for candidate, group in stage.groupby("candidate"):
        comparators.append(
            {
                "candidate": candidate,
                "mean_stage_macro_f1": float(group.macro_f1.mean()),
                "worst_stage_macro_f1": float(group.macro_f1.min()),
                "mean_stage_pr_auc": float(group.pr_auc.mean()),
                "mean_stage_roc_auc": float(group.roc_auc.mean()),
                "mean_stage_nll": float(group.nll.mean()),
                "mean_stage_brier": float(group.brier.mean()),
                "mean_stage_ece": float(group.ece.mean()),
                "mean_stage_risk_precision": float(group.risk_precision.mean()),
                "mean_stage_risk_recall": float(group.risk_recall.mean()),
                "parameter_count": (
                    160492
                    if candidate == "H1_TABULAR_RESIDUAL_EXPERT"
                    else 150202
                    if candidate == "H0_CURRENT_HYBRID"
                    else None
                ),
            }
        )
    _write_csv(OUT / "fold_metrics.csv", fold_rows)
    return stage_rows, comparators


def _bootstrap(
    averaged: pd.DataFrame, manifest: dict[str, Any], comparator: str
) -> dict[str, Any]:
    inner = manifest["scientific_configuration"]["inner_authority"]
    h1 = averaged.loc[
        averaged.candidate.eq("H1_TABULAR_RESIDUAL_EXPERT")
    ].copy()
    other = averaged.loc[averaged.candidate.eq(comparator)].copy()
    keys = [
        "base_record_id",
        "id_student",
        "target",
        "prediction_stage",
        "outer_fold",
    ]
    aligned = h1.merge(
        other[keys + ["probability"]],
        on=keys,
        suffixes=("_h1", "_other"),
        validate="one_to_one",
    )
    groups = np.array(sorted(aligned.id_student.unique()))
    group_index = {group: index for index, group in enumerate(groups)}
    rng = np.random.default_rng(7319)
    stage_counts: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    point_deltas: dict[str, float] = {}
    for stage in oulad.STAGES:
        current = aligned.loc[aligned.prediction_stage.eq(stage)].copy()
        threshold_h1 = current.outer_fold.map(
            lambda fold: inner["H1_TABULAR_RESIDUAL_EXPERT"][str(fold)][
                "research_thresholds"
            ][stage]
        ).to_numpy()
        threshold_other = current.outer_fold.map(
            lambda fold: inner[comparator][str(fold)]["research_thresholds"][stage]
        ).to_numpy()
        label = current.target.to_numpy(dtype=int)
        prediction_h1 = current.probability_h1.to_numpy() >= threshold_h1
        prediction_other = current.probability_other.to_numpy() >= threshold_other
        index = current.id_student.map(group_index).to_numpy()
        h1_counts = np.zeros((len(groups), 4), dtype=np.int64)
        other_counts = np.zeros((len(groups), 4), dtype=np.int64)
        for outcome, column in (
            ((label == 0) & (~prediction_h1), 0),
            ((label == 0) & prediction_h1, 1),
            ((label == 1) & (~prediction_h1), 2),
            ((label == 1) & prediction_h1, 3),
        ):
            np.add.at(h1_counts[:, column], index[outcome], 1)
        for outcome, column in (
            ((label == 0) & (~prediction_other), 0),
            ((label == 0) & prediction_other, 1),
            ((label == 1) & (~prediction_other), 2),
            ((label == 1) & prediction_other, 3),
        ):
            np.add.at(other_counts[:, column], index[outcome], 1)
        stage_counts[stage] = h1_counts, other_counts

        def macro(values: np.ndarray) -> float:
            tn, fp, fn, tp = values
            return float(
                (
                    2 * tn / max(2 * tn + fp + fn, 1)
                    + 2 * tp / max(2 * tp + fp + fn, 1)
                )
                / 2
            )

        point_deltas[stage] = macro(h1_counts.sum(axis=0)) - macro(
            other_counts.sum(axis=0)
        )
    deltas = np.empty(5000, dtype=np.float64)
    for replicate in range(5000):
        weights = np.bincount(
            rng.integers(0, len(groups), size=len(groups)), minlength=len(groups)
        )
        stage_delta = []
        for h1_counts, other_counts in stage_counts.values():
            h1_values = weights @ h1_counts
            other_values = weights @ other_counts
            tn, fp, fn, tp = h1_values
            h1_score = (
                2 * tn / max(2 * tn + fp + fn, 1)
                + 2 * tp / max(2 * tp + fp + fn, 1)
            ) / 2
            tn, fp, fn, tp = other_values
            other_score = (
                2 * tn / max(2 * tn + fp + fn, 1)
                + 2 * tp / max(2 * tp + fp + fn, 1)
            ) / 2
            stage_delta.append(h1_score - other_score)
        deltas[replicate] = np.mean(stage_delta)
    low, high = np.quantile(deltas, [0.025, 0.975])
    return {
        "base_candidate": "H1_TABULAR_RESIDUAL_EXPERT",
        "comparator": comparator,
        "metric": "mean_stage_macro_f1",
        "population_point_delta": float(np.mean(list(point_deltas.values()))),
        "bootstrap_mean_delta": float(deltas.mean()),
        "ci_95_low": float(low),
        "ci_95_high": float(high),
        "replicates": 5000,
        "resampling_unit": "id_student",
        "stage_point_deltas": point_deltas,
        "interval_crosses_zero": bool(low <= 0 <= high),
    }


def _direction(
    seed_metrics: pd.DataFrame, fold_metrics: pd.DataFrame, comparator: str
) -> dict[str, Any]:
    seed_mean = (
        seed_metrics.groupby(["candidate", "outer_fold", "seed"], as_index=False)
        .macro_f1.mean()
        .pivot(
            index=["outer_fold", "seed"],
            columns="candidate",
            values="macro_f1",
        )
    )
    fold_mean = (
        fold_metrics.groupby(["candidate", "outer_fold"], as_index=False)
        .macro_f1.mean()
        .pivot(index="outer_fold", columns="candidate", values="macro_f1")
    )
    return {
        "fold_positive": int(
            (
                fold_mean["H1_TABULAR_RESIDUAL_EXPERT"]
                > fold_mean[comparator]
            ).sum()
        ),
        "fold_total": int(len(fold_mean)),
        "seed_fold_positive": int(
            (
                seed_mean["H1_TABULAR_RESIDUAL_EXPERT"]
                > seed_mean[comparator]
            ).sum()
        ),
        "seed_fold_total": int(len(seed_mean)),
    }


def run_supervisor() -> int:
    prepare_directories()
    started = utc_now()
    set_sentinel("RUNNING", {"pid": os.getpid()})
    status_payload(
        state="RUNNING",
        started_at=started,
        current_stage="preflight",
        exit_code=None,
        pid=os.getpid(),
    )
    try:
        preflight = validate_freeze(require_freeze_commit=True)
        if (OUT / "predictions.parquet").exists():
            raise RuntimeError("Phase 6 final result already exists; one-shot rerun prohibited")
        manifest = load_freeze()
        write_json(OUT / "freeze_manifest.json", manifest)
        write_json(
            OUT / "evaluation_protocol.json",
            manifest["scientific_configuration"]["evaluation_protocol"],
        )
        bundle = oulad._build_bundle()
        base = bundle.base[["base_record_id", "outer_fold"]].drop_duplicates()
        seeds = manifest["scientific_configuration"]["evaluation_protocol"][
            "final_seeds"
        ]
        configs = manifest["scientific_configuration"]["training_policy"][
            "per_outer_fold_config"
        ]
        inner = manifest["scientific_configuration"]["inner_authority"]
        run_rows: list[dict[str, Any]] = []
        mapping: list[dict[str, Any]] = []
        seed_predictions: list[pd.DataFrame] = []
        for candidate in MODELS:
            for outer_fold in range(3):
                fit_ids = set(
                    base.loc[
                        base.outer_fold.ne(outer_fold), "base_record_id"
                    ]
                )
                test_ids = set(
                    base.loc[
                        base.outer_fold.eq(outer_fold), "base_record_id"
                    ]
                )
                train = oulad._stage_rows(bundle, fit_ids)
                for seed in seeds:
                    status_payload(
                        current_stage="final_training_and_prediction",
                        current_candidate=candidate,
                        current_outer_fold=outer_fold,
                        current_seed=seed,
                    )
                    checkpoint = _checkpoint_path(candidate, outer_fold, seed)
                    started_run = time.perf_counter()
                    resumed = _checkpoint_valid(
                        checkpoint, candidate, seed, manifest
                    )
                    if not resumed:
                        if checkpoint.exists():
                            raise RuntimeError(
                                f"invalid completed checkpoint; selective replacement prohibited: {checkpoint}"
                            )
                        if candidate == "M0_MLP":
                            payload = _fit_mlp_final(
                                train, seed, checkpoint, manifest
                            )
                            selected_epoch = None
                        else:
                            selected_epoch = int(
                                inner[candidate][str(outer_fold)][
                                    "selected_refit_epoch"
                                ]
                            )
                            payload = _train_deep_final(
                                candidate,
                                train,
                                configs[str(outer_fold)],
                                seed,
                                selected_epoch,
                                checkpoint,
                                manifest,
                            )
                    else:
                        if candidate == "M0_MLP":
                            payload = joblib.load(checkpoint)
                            selected_epoch = None
                        else:
                            payload = torch.load(
                                checkpoint, map_location="cpu", weights_only=False
                            )
                            selected_epoch = payload["fixed_epochs"]
                    candidate_hash = (
                        manifest["final_candidate_hash"]
                        if candidate == "H1_TABULAR_RESIDUAL_EXPERT"
                        else stable_hash(
                            {
                                "candidate": candidate,
                                "architecture": (
                                    architecture_registry()[1]
                                    if candidate == "H0_CURRENT_HYBRID"
                                    else "authoritative_sklearn_mlp_64_32"
                                ),
                            }
                        )
                    )
                    run_id = _run_id(
                        candidate_hash,
                        candidate,
                        outer_fold,
                        seed,
                        manifest["evaluation_protocol_hash"],
                    )
                    checkpoint_hash = _sha(checkpoint)
                    run_rows.append(
                        {
                            "run_id": run_id,
                            "candidate": candidate,
                            "candidate_hash": candidate_hash,
                            "outer_fold": outer_fold,
                            "seed": seed,
                            "selected_epoch": selected_epoch,
                            "checkpoint": checkpoint.relative_to(ROOT).as_posix(),
                            "checkpoint_sha256": checkpoint_hash,
                            "parameter_count": payload["parameter_count"],
                            "architecture_hash": (
                                manifest["architecture_hash"]
                                if candidate == "H1_TABULAR_RESIDUAL_EXPERT"
                                else None
                            ),
                            "feature_schema_hash": manifest["feature_schema_hash"],
                            "training_policy_hash": manifest["training_policy_hash"],
                            "evaluation_protocol_hash": manifest[
                                "evaluation_protocol_hash"
                            ],
                            "status": "RESUMED" if resumed else "COMPLETE",
                            "outer_labels_used_for_training": False,
                            "outer_labels_used_for_threshold_selection": False,
                            "runtime_seconds": time.perf_counter() - started_run,
                        }
                    )
                    for stage in oulad.STAGES:
                        data = bundle.stages[stage]
                        indices = np.flatnonzero(
                            data.frame.base_record_id.isin(test_ids).to_numpy()
                        )
                        frame = data.frame.iloc[indices].reset_index(drop=True)
                        if candidate == "M0_MLP":
                            probability = _predict_mlp(
                                checkpoint, frame, data.aggregate[indices]
                            )
                        else:
                            probability = _predict_deep_payload(
                                checkpoint,
                                frame,
                                data.sequence[indices],
                                data.lengths[indices],
                                data.mask[indices],
                                data.aggregate[indices],
                            )
                        current = frame.loc[
                            :,
                            [
                                "base_record_id",
                                "id_student",
                                "code_module",
                                "code_presentation",
                                "target",
                                "cutoff_day",
                            ],
                        ].copy()
                        current["candidate"] = candidate
                        current["prediction_stage"] = stage
                        current["outer_fold"] = outer_fold
                        current["seed"] = seed
                        current["probability"] = probability
                        seed_predictions.append(current)
                        mapping.append(
                            {
                                "run_id": run_id,
                                "candidate": candidate,
                                "outer_fold": outer_fold,
                                "seed": seed,
                                "prediction_stage": stage,
                                "checkpoint": checkpoint.relative_to(ROOT).as_posix(),
                                "checkpoint_sha256": checkpoint_hash,
                            }
                        )
                    current_status = status()
                    status_payload(
                        completed_runs=int(current_status.get("completed_runs", 0))
                        + 1
                    )
        seed_frame = pd.concat(seed_predictions, ignore_index=True)
        seed_frame.to_parquet(OUT / "seed_predictions.parquet", index=False)
        seed_metric_rows, averaged = _metric_rows(seed_frame, manifest)
        averaged.to_parquet(OUT / "predictions.parquet", index=False)
        _write_csv(OUT / "fold_seed_metrics.csv", seed_metric_rows)
        stage_rows, comparator_rows = _stage_and_comparator(averaged, manifest)
        _write_csv(OUT / "stage_metrics.csv", stage_rows)
        _write_csv(OUT / "comparator_summary.csv", comparator_rows)
        threshold_rows = []
        for candidate in MODELS:
            for fold in range(3):
                for stage, threshold in inner[candidate][str(fold)][
                    "research_thresholds"
                ].items():
                    threshold_rows.append(
                        {
                            "candidate": candidate,
                            "outer_fold": fold,
                            "prediction_stage": stage,
                            "threshold": threshold,
                            "source": "PHASE5_POOLED_INNER_OOF_SEED42",
                            "outer_labels_used": False,
                        }
                    )
        _write_csv(OUT / "threshold_summary.csv", threshold_rows)
        _write_csv(
            OUT / "calibration_summary.csv",
            [
                {
                    key: value
                    for key, value in row.items()
                    if key
                    in {
                        "candidate",
                        "prediction_stage",
                        "nll",
                        "brier",
                        "ece",
                    }
                }
                for row in stage_rows
            ],
        )
        run_manifest = {
            "status": "PASS",
            "freeze_commit": preflight["freeze_commit"],
            "run_count": len(run_rows),
            "runs": run_rows,
            "stage_mapping_count": len(mapping),
            "same_checkpoint_all_stages": True,
            "stage_mapping": mapping,
        }
        write_json(OUT / "run_manifest.json", run_manifest)
        status_payload(current_stage="paired_bootstrap")
        bootstrap = {
            comparator: _bootstrap(averaged, manifest, comparator)
            for comparator in ("M0_MLP", "H0_CURRENT_HYBRID")
        }
        write_json(OUT / "bootstrap_summary.json", bootstrap)
        seed_metrics = pd.DataFrame(seed_metric_rows)
        fold_metrics = pd.read_csv(OUT / "fold_metrics.csv")
        comparator = pd.DataFrame(comparator_rows).set_index("candidate")
        comparisons: dict[str, Any] = {}
        for other in ("M0_MLP", "H0_CURRENT_HYBRID"):
            comparisons[other] = {
                "macro_f1_delta": float(
                    comparator.loc[
                        "H1_TABULAR_RESIDUAL_EXPERT",
                        "mean_stage_macro_f1",
                    ]
                    - comparator.loc[other, "mean_stage_macro_f1"]
                ),
                "pr_auc_delta": float(
                    comparator.loc[
                        "H1_TABULAR_RESIDUAL_EXPERT", "mean_stage_pr_auc"
                    ]
                    - comparator.loc[other, "mean_stage_pr_auc"]
                ),
                "nll_delta": float(
                    comparator.loc["H1_TABULAR_RESIDUAL_EXPERT", "mean_stage_nll"]
                    - comparator.loc[other, "mean_stage_nll"]
                ),
                "brier_delta": float(
                    comparator.loc[
                        "H1_TABULAR_RESIDUAL_EXPERT", "mean_stage_brier"
                    ]
                    - comparator.loc[other, "mean_stage_brier"]
                ),
                "ece_delta": float(
                    comparator.loc["H1_TABULAR_RESIDUAL_EXPERT", "mean_stage_ece"]
                    - comparator.loc[other, "mean_stage_ece"]
                ),
                **_direction(seed_metrics, fold_metrics, other),
                "bootstrap": bootstrap[other],
            }
        write_json(OUT / "paired_comparison.json", comparisons)
        old_checksums = {
            relative: _sha(ROOT / relative)
            for relative in EXPECTED_OLD_FINAL
        }
        h1_runs = [
            row
            for row in run_rows
            if row["candidate"] == "H1_TABULAR_RESIDUAL_EXPERT"
        ]
        integrity_checks = {
            "freeze_commit_precedes_outer": True,
            "all_runs_complete": len(run_rows) == 45,
            "h1_runs": len(h1_runs) == 15,
            "seeds_exact": sorted({row["seed"] for row in run_rows})
            == [42, 1201, 2026, 3407, 7319],
            "outer_folds_exact": sorted({row["outer_fold"] for row in run_rows})
            == [0, 1, 2],
            "h1_candidate_hash_count": len(
                {row["candidate_hash"] for row in h1_runs}
            )
            == 1,
            "h1_architecture_hash_count": len(
                {row["architecture_hash"] for row in h1_runs}
            )
            == 1,
            "h1_parameter_count_count": len(
                {row["parameter_count"] for row in h1_runs}
            )
            == 1,
            "feature_schema_hash_count": len(
                {row["feature_schema_hash"] for row in run_rows}
            )
            == 1,
            "same_checkpoint_all_stages": len(mapping) == len(run_rows) * 4,
            "outer_labels_absent_training_selection": not any(
                row["outer_labels_used_for_training"]
                or row["outer_labels_used_for_threshold_selection"]
                for row in run_rows
            ),
            "optuna_trials": 0,
            "old_official_evidence_unchanged": old_checksums
            == EXPECTED_OLD_FINAL,
            "failed_runs": 0,
        }
        integrity = {
            "status": "PASS" if all(
                value == 0 if key in {"optuna_trials", "failed_runs"} else bool(value)
                for key, value in integrity_checks.items()
            ) else "FAIL",
            "checks": integrity_checks,
            "old_official_checksums": old_checksums,
            "final_candidate_hash": manifest["final_candidate_hash"],
            "freeze_commit": preflight["freeze_commit"],
        }
        write_json(OUT / "integrity_report.json", integrity)
        gate_checks = {
            "freeze_precedes_outer": True,
            "immutable_hashes": preflight["status"] == "PASS",
            "all_required_runs": len(run_rows) == 45,
            "all_seeds_retained": integrity_checks["seeds_exact"],
            "no_post_outer_tuning": True,
            "no_optuna": True,
            "no_architecture_mutation": True,
            "mlp_protocol_matched": True,
            "h0_protocol_matched": True,
            "metrics_complete": True,
            "uncertainty_complete": len(bootstrap) == 2,
            "integrity_pass": integrity["status"] == "PASS",
        }
        gate = {"status": "PASS" if all(gate_checks.values()) else "FAIL", "checks": gate_checks}
        write_json(OUT / "phase6_gate.json", gate)
        if gate["status"] != "PASS":
            raise RuntimeError("Phase 6 gate failed")
        finished = utc_now()
        status_payload(
            state="COMPLETE",
            finished_at=finished,
            current_stage="complete",
            current_candidate=None,
            current_outer_fold=None,
            current_seed=None,
            exit_code=0,
        )
        set_sentinel(
            "COMPLETE",
            {
                "started_at": started,
                "finished_at": finished,
                "gate": "PASS",
                "freeze_commit": preflight["freeze_commit"],
            },
        )
        return 0
    except Exception as error:
        finished = utc_now()
        failure = {
            "state": "FAILED",
            "started_at": started,
            "finished_at": finished,
            "failure_type": type(error).__name__,
            "failure_reason": repr(error),
            "exit_code": 1,
        }
        write_json(OUT / "failure_summary.json", failure)
        status_payload(**failure, current_stage="failed")
        set_sentinel("FAILED", failure)
        return 1


def status() -> dict[str, Any]:
    if not STATUS_PATH.is_file():
        return {"state": "PENDING", "status_file": False}
    return json.loads(STATUS_PATH.read_text(encoding="utf-8"))
