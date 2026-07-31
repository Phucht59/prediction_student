"""Phase 9: inner-only H1 endpoint recipe recovery.

This module deliberately has no API that accepts outer-test labels.  It keeps
the Phase 5 H1 topology fixed while comparing the Phase 7 recipe with the
scientifically defensible subset of the historical H0 endpoint recipe.
"""

from __future__ import annotations

import copy
import csv
import hashlib
import json
import math
import os
import random
import sqlite3
import subprocess
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import optuna
import pandas as pd
import torch
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_recall_fscore_support,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from src.models._oulad import _OULADTemporalEncoder
from src.pipelines import oulad
from src.training.control import select_research_threshold, stable_hash
from src.training.phase3_optuna import _risk_loss, write_json
from src.training.phase5_mlp_gap import architecture_registry, make_model
from src.training.phase7_endpoint import (
    EARLY_WARNING_FILES,
    ENDPOINT_STAGE,
    PARAMETER_COUNT,
    _endpoint_rows,
    control_config,
)

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "artifacts" / "audit" / "phase9"
RUNTIME = OUT / "runtime"
RUNS = RUNTIME / "runs"
PREDICTIONS = RUNTIME / "predictions"
PRETRAIN = RUNTIME / "pretraining"
LOGS = OUT / "logs"
STATUS = RUNTIME / "phase9_status.json"
RUNNING = RUNTIME / "PHASE9_RUNNING"
COMPLETE = RUNTIME / "PHASE9_COMPLETE"
FAILED = RUNTIME / "PHASE9_FAILED"
PHASE8_GATE = ROOT / "artifacts" / "audit" / "phase8" / "phase8_gate.json"
PHASE8_H0 = ROOT / "artifacts" / "audit" / "phase8" / "h0_endpoint_profile.json"
PHASE8_H1 = ROOT / "artifacts" / "audit" / "phase8" / "h1_endpoint_profile.json"
SPLIT_MANIFEST = ROOT / "data" / "processed" / "study_c_oulad" / "manifests" / "split_manifest.csv"
HISTORICAL_H0_PRED = ROOT / "artifacts" / "final" / "predictions" / "cnn_bilstm_oulad" / "oof_predictions.parquet"
HISTORICAL_MLP_PRED = ROOT / "artifacts" / "final" / "teacher_feedback_validation" / "mlp_comparator" / "oulad" / "oof_predictions.parquet"

OUTER_FOLDS = (0, 1, 2)
SEARCH_SEED = 42
STABILITY_SEEDS = (1201, 2026)
H0_EPOCHS = 8
H0_PRETRAIN_EPOCHS = 5
MAX_OPTUNA_TRIALS = 24
ARCHITECTURE_ID = "H1_TABULAR_RESIDUAL_EXPERT"
SCORE_DECISION = "SCORE_PROXY_REJECTED"

H0_CONFIG = {
    "learning_rate": 0.001462740013487936,
    "weight_decay": 1.9904886022582856e-07,
    "dropout": 0.28548926103262684,
    "batch_size": 256,
    "loss_policy": "standard_bce",
    "pos_weight_strategy": "not_applicable",
    "survival_weight": 0.15,
    "outcome_weight": 0.15,
}

LEGAL_PRETRAIN_TASKS = (
    "masked_activity_band",
    "masked_active_state",
    "masked_submission_band",
    "masked_inactivity_transition",
    "next_active_state",
    "next_activity_direction",
    "next_submission_band",
    "next_inactivity_transition",
)
REJECTED_PRETRAIN_TASKS = (
    "masked_score_availability",
    "next_score_state_transition",
)

COMPACT_SUMMARIES: dict[str, tuple[str, ...]] = {
    "total_clicks": ("sum", "mean", "last", "slope", "recent_2_week_mean"),
    "active_days": ("sum", "mean", "last", "slope", "recent_2_week_mean"),
    "unique_sites": ("mean", "last", "recent_2_week_mean"),
    "unique_activity_types": ("mean", "last", "recent_2_week_mean"),
    "content_clicks": ("sum", "slope", "recent_2_week_mean"),
    "forum_clicks": ("sum", "slope", "recent_2_week_mean"),
    "quiz_clicks": ("sum", "slope", "recent_2_week_mean"),
    "assessment_related_clicks": ("sum", "slope", "recent_2_week_mean"),
    "submitted_assessment_count": ("sum", "last"),
    "late_submission_count": ("sum", "last"),
    "available_score_count": ("sum", "last"),
    "cumulative_mean_score": ("last", "slope", "recent_2_week_mean"),
    "cumulative_weighted_score": ("last", "slope", "recent_2_week_mean"),
    "days_since_last_vle_activity": ("last", "slope", "recent_2_week_mean"),
    "weeks_without_activity": ("sum", "last", "recent_2_week_mean"),
    "score_missing_mask": ("sum", "last"),
}
SCORE_CHANNELS = {
    "available_score_count",
    "cumulative_mean_score",
    "cumulative_weighted_score",
    "score_missing_mask",
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        rows = [{"status": "NOT_TRIGGERED"}]
    fields = list(dict.fromkeys(key for row in rows for key in row))
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: json.dumps(value, sort_keys=True) if isinstance(value, (dict, list)) else value for key, value in row.items()})
    temporary.replace(path)


def _directories() -> None:
    for path in (OUT, RUNTIME, RUNS, PREDICTIONS, PRETRAIN, LOGS):
        path.mkdir(parents=True, exist_ok=True)


def _status(**updates: Any) -> dict[str, Any]:
    payload = json.loads(STATUS.read_text(encoding="utf-8")) if STATUS.is_file() else {
        "state": "PENDING", "started_at": None, "finished_at": None,
        "current_stage": "audit", "completed_runs": 0, "failed_runs": 0,
        "current_candidate": None, "optuna_triggered": False, "exit_code": None,
        "pid": os.getpid(),
    }
    payload.update(updates)
    write_json(STATUS, payload)
    return payload


def _sentinel(state: str, **details: Any) -> None:
    for path in (RUNNING, COMPLETE, FAILED):
        if path.exists():
            path.unlink()
    target = {"RUNNING": RUNNING, "COMPLETE": COMPLETE, "FAILED": FAILED}[state]
    write_json(target, {"state": state, "at": utc_now(), **details})


def early_warning_checksums() -> dict[str, str]:
    return {relative: _sha(ROOT / relative) for relative in EARLY_WARNING_FILES}


def architecture_identity() -> dict[str, Any]:
    row = next(item for item in architecture_registry() if item["architecture_id"] == ARCHITECTURE_ID)
    if int(row["parameter_count"]) != PARAMETER_COUNT:
        raise RuntimeError("H1 parameter count changed")
    return row


def score_feature_authority() -> dict[str, Any]:
    decision = {
        "status": "PASS",
        "decision": SCORE_DECISION,
        "endpoint": "F2_MIDDLE: floor(module_presentation_length * 0.50)",
        "known_event_rule": "0 <= event_day < cutoff_day",
        "historical_proxy": "max(date_submitted, assessment_due_date) < cutoff_day",
        "score_release_timestamp_present_in_raw_oulad": False,
        "reason": "OULAD records submission and assessment dates but not the date a marked score became visible. The historical proxy is cutoff-consistent on known dates but cannot prove score availability at prediction time.",
        "performance_used_to_authorize": False,
        "score_features_used_in_h1r": False,
        "rejected_sequence_channels": sorted(SCORE_CHANNELS),
        "rejected_pretraining_tasks": list(REJECTED_PRETRAIN_TASKS),
        "evidence": [
            "src/pipelines/oulad.py: score values explicitly excluded",
            "reports/final/OULAD_CUTOFF_AUDIT.md",
            "artifacts/audit/phase8/h0_endpoint_profile.json",
            "OULAD studentAssessment schema: date_submitted and score, no score_release_date",
        ],
    }
    write_json(OUT / "score_feature_authority.json", decision)
    return decision


def holdout_availability_audit() -> dict[str, Any]:
    split = pd.read_csv(SPLIT_MANIFEST)
    role_counts = {str(key): int(value) for key, value in split.groupby("role").size().items()}
    future = split.loc[split.role.eq("future_candidate")]
    current_paths = sorted(
        path.relative_to(ROOT).as_posix()
        for root in (ROOT / "artifacts", ROOT / "reports")
        for path in root.rglob("*")
        if path.is_file() and any(token in path.as_posix().lower() for token in ("future_presentation", "future_candidate"))
    )
    historical_output = subprocess.check_output(
        ["git", "log", "--all", "--name-only", "--pretty=format:", "--", "artifacts", "reports"],
        cwd=ROOT,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    historical_paths = sorted(
        {
            line.strip()
            for line in historical_output.splitlines()
            if "future_presentation" in line.lower()
        }
    )
    prior_paths = sorted(set(current_paths) | set(historical_paths))
    payload = {
        "status": "PASS",
        "new_untouched_holdout_available": False,
        "reason": "All historical-development records were repeatedly evaluated, and the only named future_candidate population was already scored/reported by historical Future-presentation studies. Resplitting observed OULAD cannot restore untouched status.",
        "role_row_counts": role_counts,
        "future_candidate_rows": int(len(future)),
        "future_candidate_unique_records": int(future.record_id.nunique()) if len(future) else 0,
        "prior_future_evidence_count": len(prior_paths),
        "prior_future_evidence_examples": prior_paths[:30],
        "history_scan": "git log --all --name-only over artifacts and reports",
        "phase7_outer_reusable_as_untouched": False,
        "random_resplit_is_new_holdout": False,
        "confirmation_allowed": False,
    }
    write_json(OUT / "holdout_availability_audit.json", payload)
    write_json(OUT / "new_holdout_manifest.json", {
        "status": "NOT_AVAILABLE",
        "record_ids": None,
        "labels_accessed": False,
        "reason": payload["reason"],
    })
    return payload


def _inner_splits(bundle: oulad.Bundle, outer_fold: int) -> list[tuple[set[str], set[str]]]:
    base = bundle.stages[ENDPOINT_STAGE].frame.loc[:, ["base_record_id", "id_student", "outer_fold", "target"]].drop_duplicates("base_record_id")
    train = base.loc[base.outer_fold.ne(outer_fold)].reset_index(drop=True)
    splitter = StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=3407 + outer_fold)
    return [
        (set(train.iloc[fit].base_record_id), set(train.iloc[val].base_record_id))
        for fit, val in splitter.split(train, train.target, train.id_student)
    ]


def _masked_sequence_stats(sequence: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    valid = mask.astype(np.float32)
    count = max(1.0, float(valid.sum()))
    mean = (sequence * valid[..., None]).sum(axis=(0, 1)) / count
    variance = (((sequence - mean) ** 2) * valid[..., None]).sum(axis=(0, 1)) / count
    scale = np.sqrt(np.maximum(variance, 1e-12)).clip(1e-6)
    return mean.astype(np.float32), scale.astype(np.float32)


def _normalize_sequence(sequence: np.ndarray, mask: np.ndarray, mean: np.ndarray, scale: np.ndarray) -> np.ndarray:
    result = ((sequence - mean) / scale).astype(np.float32)
    result *= mask[..., None].astype(np.float32)
    return result


def _summary(values: np.ndarray, mask: np.ndarray, operation: str) -> np.ndarray:
    valid = mask.astype(bool)
    lengths = valid.sum(axis=1).clip(min=1)
    if operation == "sum":
        return (values * valid).sum(axis=1)
    if operation == "mean":
        return (values * valid).sum(axis=1) / lengths
    last = values[np.arange(len(values)), lengths - 1]
    if operation == "last":
        return last
    if operation == "recent_2_week_mean":
        return np.asarray([row[max(0, length - 2):length].mean() for row, length in zip(values, lengths)])
    if operation == "slope":
        slopes = []
        for row, length in zip(values, lengths):
            if length < 2:
                slopes.append(0.0)
            else:
                x = np.arange(length, dtype=np.float64)
                slopes.append(float(np.polyfit(x, row[:length], 1)[0]))
        return np.asarray(slopes)
    raise ValueError(operation)


def valid_h0_aggregate(sequence: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Map the valid, score-free H0 compact schema into H1's frozen 165 inputs."""
    order = list(oulad.CHANNELS)
    columns: list[np.ndarray] = []
    for channel, operations in COMPACT_SUMMARIES.items():
        values = sequence[:, :, order.index(channel)]
        for operation in operations:
            value = np.zeros(len(sequence), dtype=np.float32) if channel in SCORE_CHANNELS else _summary(values, mask, operation).astype(np.float32)
            columns.append(value)
    inactivity = sequence[:, :, order.index("weeks_without_activity")]
    columns.append(((inactivity > 0) & mask.astype(bool)).sum(axis=1).astype(np.float32))
    compact = np.column_stack(columns).astype(np.float32)
    if compact.shape[1] != 49:
        raise RuntimeError(f"historical compact schema changed: {compact.shape}")
    padded = np.zeros((len(compact), 165), dtype=np.float32)
    padded[:, :49] = compact
    return padded


class RecipePreprocessor:
    def __init__(self, recipe: str):
        self.recipe = recipe

    def fit(self, frame: pd.DataFrame, aggregate: np.ndarray, sequence: np.ndarray, mask: np.ndarray) -> "RecipePreprocessor":
        self.deep = oulad._DeepPreprocessor().fit(frame, aggregate)
        self.sequence_mean, self.sequence_scale = _masked_sequence_stats(sequence, mask)
        return self

    def transform(self, frame: pd.DataFrame, aggregate: np.ndarray, sequence: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        transformed_aggregate, static = self.deep.transform(frame, aggregate)
        transformed_sequence = sequence.astype(np.float32)
        if self.recipe not in {"A0_PHASE7_H1_CONTROL", "B2_WITHOUT_H0_PREPROCESSING"}:
            transformed_sequence = _normalize_sequence(sequence, mask, self.sequence_mean, self.sequence_scale)
        return transformed_sequence, transformed_aggregate, static


class LegalTemporalPretrainer(nn.Module):
    def __init__(self, config: dict[str, Any]):
        super().__init__()
        model = make_model(ARCHITECTURE_ID, 165, 13, config)
        self.encoder = model.backbone.temporal
        width = self.encoder.sequence_output_dim
        self.activity_band = nn.Linear(width, 3)
        self.active = nn.Linear(width, 1)
        self.submission_band = nn.Linear(width, 3)
        self.inactivity_transition = nn.Linear(width, 1)
        self.next_active = nn.Linear(width, 1)
        self.activity_direction = nn.Linear(width, 3)
        self.next_submission_band = nn.Linear(width, 3)
        self.next_inactivity_transition = nn.Linear(width, 1)

    def forward(self, sequence: torch.Tensor, lengths: torch.Tensor, mask: torch.Tensor) -> dict[str, torch.Tensor]:
        values = self.encoder.encode_sequence(sequence, lengths, mask)
        return {
            "activity_band": self.activity_band(values),
            "active": self.active(values).squeeze(-1),
            "submission_band": self.submission_band(values),
            "inactivity_transition": self.inactivity_transition(values).squeeze(-1),
            "next_active": self.next_active(values).squeeze(-1),
            "activity_direction": self.activity_direction(values),
            "next_submission_band": self.next_submission_band(values),
            "next_inactivity_transition": self.next_inactivity_transition(values).squeeze(-1),
        }


def _pretrain_labels(raw: np.ndarray, mask: np.ndarray) -> dict[str, np.ndarray]:
    order = list(oulad.CHANNELS)
    clicks = raw[:, :, order.index("total_clicks")]
    submissions = raw[:, :, order.index("submitted_assessment_count")]
    inactivity = raw[:, :, order.index("weeks_without_activity")]
    positive = clicks[(mask > 0) & (clicks > 0)]
    lower, upper = np.quantile(positive, [0.33, 0.66]) if len(positive) else (1.0, 2.0)
    activity_band = np.where(clicks <= 0, 0, np.where(clicks <= lower, 1, 2)).astype(np.int64)
    submission_band = np.where(submissions <= 0, 0, np.where(submissions < 2, 1, 2)).astype(np.int64)
    active = (clicks > 0).astype(np.float32)
    transition = np.zeros_like(active)
    transition[:, 1:] = (inactivity[:, 1:] > inactivity[:, :-1]).astype(np.float32)
    direction = np.ones_like(activity_band)
    direction[:, :-1] = np.where(clicks[:, 1:] < clicks[:, :-1], 0, np.where(clicks[:, 1:] > clicks[:, :-1], 2, 1))
    return {"activity_band": activity_band, "active": active, "submission_band": submission_band, "inactivity_transition": transition, "activity_direction": direction}


def _mask_weeks(valid: np.ndarray, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    selected = np.zeros_like(valid, dtype=bool)
    for row, row_valid in enumerate(valid.astype(bool)):
        positions = np.flatnonzero(row_valid)
        if len(positions) > 1:
            selected[row, rng.choice(positions, size=min(2, max(1, len(positions) // 10)), replace=False)] = True
    return selected


def fit_legal_pretraining(sequence: np.ndarray, raw_sequence: np.ndarray, lengths: np.ndarray, mask: np.ndarray, config: dict[str, Any], seed: int) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    device = torch.device("cuda")
    model = LegalTemporalPretrainer(config).to(device)
    labels = _pretrain_labels(raw_sequence, mask)
    dataset = TensorDataset(torch.from_numpy(sequence), torch.from_numpy(lengths.astype(np.int64)), torch.from_numpy(mask.astype(np.float32)), *(torch.from_numpy(value) for value in labels.values()))
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=float(config["weight_decay"]))
    histories = []
    label_names = list(labels)
    best_loss = float("inf"); best_state = copy.deepcopy(model.state_dict())
    for epoch in range(1, H0_PRETRAIN_EPOCHS + 1):
        loader = DataLoader(dataset, batch_size=int(config["batch_size"]), shuffle=True, num_workers=0, generator=torch.Generator().manual_seed(seed + epoch))
        losses = []; offset = 0
        model.train()
        for batch in loader:
            seq, lens, valid = batch[:3]
            target = {name: batch[index + 3].to(device) for index, name in enumerate(label_names)}
            selected = _mask_weeks(valid.numpy(), seed + epoch * 1_000_003 + offset); offset += len(seq)
            corrupted = seq.clone(); corrupted[torch.from_numpy(selected)] = 0
            output = model(corrupted.to(device), lens.to(device), valid.to(device))
            selected_t = torch.from_numpy(selected).to(device)
            next_valid = (valid[:, :-1].bool() & valid[:, 1:].bool()).to(device)
            parts = [
                nn.functional.cross_entropy(output["activity_band"][selected_t], target["activity_band"][selected_t]),
                nn.functional.binary_cross_entropy_with_logits(output["active"][selected_t], target["active"][selected_t]),
                nn.functional.cross_entropy(output["submission_band"][selected_t], target["submission_band"][selected_t]),
                nn.functional.binary_cross_entropy_with_logits(output["inactivity_transition"][selected_t], target["inactivity_transition"][selected_t]),
                nn.functional.binary_cross_entropy_with_logits(output["next_active"][:, :-1][next_valid], target["active"][:, 1:][next_valid]),
                nn.functional.cross_entropy(output["activity_direction"][:, :-1][next_valid], target["activity_direction"][:, :-1][next_valid]),
                nn.functional.cross_entropy(output["next_submission_band"][:, :-1][next_valid], target["submission_band"][:, 1:][next_valid]),
                nn.functional.binary_cross_entropy_with_logits(output["next_inactivity_transition"][:, :-1][next_valid], target["inactivity_transition"][:, 1:][next_valid]),
            ]
            loss = torch.stack(parts).mean()
            optimizer.zero_grad(set_to_none=True); loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 1.0); optimizer.step()
            losses.append(float(loss.detach()))
        mean_loss = float(np.mean(losses)); histories.append({"epoch": epoch, "loss": mean_loss})
        if mean_loss < best_loss:
            best_loss = mean_loss; best_state = copy.deepcopy(model.state_dict())
    temporal = {name[len("encoder."):]: value.detach().cpu().clone() for name, value in best_state.items() if name.startswith("encoder.")}
    return temporal, {"tasks": list(LEGAL_PRETRAIN_TASKS), "rejected_tasks": list(REJECTED_PRETRAIN_TASKS), "epochs": H0_PRETRAIN_EPOCHS, "history": histories, "best_loss": best_loss, "train_only": True}


def _metrics(labels: np.ndarray, probability: np.ndarray) -> dict[str, Any]:
    threshold = select_research_threshold(labels, probability)
    predicted = probability >= float(threshold["threshold"])
    tn, fp, fn, tp = confusion_matrix(labels, predicted, labels=[0, 1]).ravel()
    macro_p, macro_r, macro_f1, _ = precision_recall_fscore_support(labels, predicted, average="macro", zero_division=0)
    risk_p, risk_r, risk_f1, _ = precision_recall_fscore_support(labels, predicted, average="binary", zero_division=0)
    bins = np.linspace(0, 1, 11); ece = 0.0
    for low, high in zip(bins[:-1], bins[1:]):
        chosen = (probability >= low) & (probability < high if high < 1 else probability <= high)
        if chosen.any():
            ece += chosen.mean() * abs(labels[chosen].mean() - probability[chosen].mean())
    return {
        "accuracy": float(accuracy_score(labels, predicted)),
        "balanced_accuracy": float(balanced_accuracy_score(labels, predicted)),
        "macro_precision": float(macro_p), "macro_recall": float(macro_r), "macro_f1": float(macro_f1),
        "pr_auc": float(average_precision_score(labels, probability)), "roc_auc": float(roc_auc_score(labels, probability)),
        "nll": float(log_loss(labels, np.clip(probability, 1e-7, 1 - 1e-7), labels=[0, 1])),
        "brier": float(brier_score_loss(labels, probability)), "ece": float(ece),
        "risk_precision": float(risk_p), "risk_recall": float(risk_r), "risk_f1": float(risk_f1),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp), "research_threshold": threshold,
    }


@dataclass
class FitResult:
    probability: np.ndarray
    selected_epoch: int
    pretraining: dict[str, Any]
    residual_alpha: float


def _predict_model(
    model: nn.Module,
    sequence: np.ndarray,
    lengths: np.ndarray,
    mask: np.ndarray,
    aggregate: np.ndarray,
    static: np.ndarray,
    *,
    disable_residual: bool,
    disable_temporal: bool,
) -> np.ndarray:
    model.eval()
    values: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(sequence), 512):
            selected = slice(start, start + 512)
            output = model(
                torch.from_numpy(sequence[selected]).to("cuda"),
                torch.from_numpy(lengths[selected].astype(np.int64)).to("cuda"),
                torch.from_numpy(mask[selected].astype(np.float32)).to("cuda"),
                torch.from_numpy(aggregate[selected]).to("cuda"),
                torch.from_numpy(static[selected]).to("cuda"),
                disable_temporal=disable_temporal,
                disable_tabular_residual=disable_residual,
            )
            values.append(torch.sigmoid(output["binary_logit"]).cpu().numpy())
    return np.concatenate(values)


def _fit(train: tuple, validation: tuple, *, recipe: str, config: dict[str, Any], seed: int, disable_residual: bool = False, disable_temporal: bool = False) -> FitResult:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    frame, raw_sequence, lengths, mask, raw_aggregate, labels, weights = train
    vf, val_raw_sequence, val_lengths, val_mask, val_raw_aggregate, val_labels, _ = validation
    use_h0_features = recipe != "A0_PHASE7_H1_CONTROL"
    aggregate = valid_h0_aggregate(raw_sequence, mask) if use_h0_features else raw_aggregate
    val_aggregate = valid_h0_aggregate(val_raw_sequence, val_mask) if use_h0_features else val_raw_aggregate
    pre = RecipePreprocessor(recipe).fit(frame, aggregate, raw_sequence, mask)
    sequence, aggregate, static = pre.transform(frame, aggregate, raw_sequence, mask)
    val_sequence, val_aggregate, val_static = pre.transform(vf, val_aggregate, val_raw_sequence, val_mask)
    model = make_model(ARCHITECTURE_ID, aggregate.shape[1], static.shape[1], config).to("cuda")
    if sum(p.numel() for p in model.parameters() if p.requires_grad) != PARAMETER_COUNT:
        raise RuntimeError("architecture or parameter count changed")
    pretraining: dict[str, Any] = {"executed": False, "reason": "Phase7 control"}
    if recipe not in {"A0_PHASE7_H1_CONTROL", "B3_WITHOUT_PRETRAINING"}:
        pretrain_key = stable_hash(
            {
                "record_ids": sorted(frame.base_record_id.astype(str).tolist()),
                "recipe": recipe,
                "seed": seed,
                "config": config,
                "tasks": LEGAL_PRETRAIN_TASKS,
            }
        )
        pretrain_path = PRETRAIN / f"{pretrain_key}.pt"
        if pretrain_path.is_file():
            cached = torch.load(pretrain_path, map_location="cpu", weights_only=False)
            temporal_state = cached["temporal_state"]
            details = {**cached["details"], "resumed": True}
        else:
            temporal_state, details = fit_legal_pretraining(
                sequence, raw_sequence, lengths, mask, config, seed
            )
            torch.save(
                {"temporal_state": temporal_state, "details": details},
                pretrain_path,
            )
            details = {**details, "resumed": False}
        missing, unexpected = model.backbone.temporal.load_state_dict(temporal_state, strict=False)
        if missing or unexpected:
            raise RuntimeError(f"pretraining state mismatch missing={missing} unexpected={unexpected}")
        pretraining = {"executed": True, **details}
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(config["learning_rate"]), weight_decay=float(config["weight_decay"]))
    risk_loss, _ = _risk_loss(labels, config, torch.device("cuda"))
    dataset = TensorDataset(torch.from_numpy(sequence), torch.from_numpy(lengths.astype(np.int64)), torch.from_numpy(mask.astype(np.float32)), torch.from_numpy(aggregate), torch.from_numpy(static), torch.from_numpy(labels.astype(np.float32)), torch.from_numpy(weights.astype(np.float32)), torch.from_numpy(frame.outcome_aux.to_numpy(dtype=np.int64)), torch.from_numpy(frame.cutoff_day.to_numpy(dtype=np.int64)), torch.from_numpy(frame.module_presentation_length.to_numpy(dtype=np.int64)), torch.from_numpy(frame.date_unregistration.fillna(-1).to_numpy(dtype=np.int64)))
    loader = DataLoader(dataset, batch_size=int(config["batch_size"]), shuffle=True, num_workers=0, generator=torch.Generator().manual_seed(seed))
    fixed = recipe not in {"A0_PHASE7_H1_CONTROL", "B4_WITHOUT_H0_TRAINING_POLICY"}
    epoch_limit = H0_EPOCHS if fixed else 15
    best_nll = float("inf"); best_state = None; best_epoch = 1; wait = 0
    for epoch in range(1, epoch_limit + 1):
        model.train()
        for batch in loader:
            seq, lens, valid, agg, stat, target, sample_weight, outcome, cutoff, course_end, unreg = (value.to("cuda") for value in batch)
            output = model(seq, lens, valid, agg, stat)
            loss, _ = oulad._multitask_loss(output, target, sample_weight, outcome, cutoff, course_end, unreg, risk_loss, survival_weight=float(config["survival_weight"]), outcome_weight=float(config["outcome_weight"]))
            optimizer.zero_grad(set_to_none=True); loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 1.0); optimizer.step()
        probability = _predict_model(
            model,
            val_sequence,
            val_lengths,
            val_mask,
            val_aggregate,
            val_static,
            disable_residual=False,
            disable_temporal=False,
        )
        nll = float(log_loss(val_labels, np.clip(probability, 1e-7, 1 - 1e-7), labels=[0, 1]))
        if fixed:
            best_epoch = epoch; best_state = copy.deepcopy(model.state_dict())
        elif nll < best_nll - 1e-6:
            best_nll = nll; best_epoch = epoch; best_state = copy.deepcopy(model.state_dict()); wait = 0
        else:
            wait += 1
            if wait >= 5:
                break
    if best_state is None:
        raise RuntimeError("no checkpoint state")
    model.load_state_dict(best_state)
    probability = _predict_model(
        model,
        val_sequence,
        val_lengths,
        val_mask,
        val_aggregate,
        val_static,
        disable_residual=disable_residual,
        disable_temporal=disable_temporal,
    )
    alpha = float(model.residual_alpha.detach().cpu())
    del model, optimizer, loader, dataset
    torch.cuda.empty_cache()
    return FitResult(probability, best_epoch, pretraining, alpha)


def evaluate_candidate(bundle: oulad.Bundle, *, candidate: str, recipe: str, outer_fold: int, seed: int, config: dict[str, Any], disable_residual: bool = False, disable_temporal: bool = False) -> dict[str, Any]:
    run_science = {"candidate": candidate, "recipe": recipe, "outer_fold": outer_fold, "seed": seed, "config": config, "architecture_hash": architecture_identity()["architecture_hash"], "score_decision": SCORE_DECISION, "disable_residual": disable_residual, "disable_temporal": disable_temporal}
    run_id = stable_hash(run_science)[:20]
    result_path = RUNS / f"{run_id}.json"; prediction_path = PREDICTIONS / f"{run_id}.parquet"
    if result_path.is_file() and prediction_path.is_file():
        return json.loads(result_path.read_text(encoding="utf-8"))
    predictions = []; epochs = []; pretraining = []; alphas = []
    for inner_fold, (fit_ids, val_ids) in enumerate(_inner_splits(bundle, outer_fold)):
        result = _fit(_endpoint_rows(bundle, fit_ids), _endpoint_rows(bundle, val_ids), recipe=recipe, config=config, seed=seed + inner_fold, disable_residual=disable_residual, disable_temporal=disable_temporal)
        validation = _endpoint_rows(bundle, val_ids)[0].loc[:, ["base_record_id", "id_student", "outer_fold", "target"]].copy()
        validation["probability"] = result.probability; validation["inner_fold"] = inner_fold
        predictions.append(validation); epochs.append(result.selected_epoch); pretraining.append(result.pretraining); alphas.append(result.residual_alpha)
    prediction = pd.concat(predictions, ignore_index=True)
    metrics = _metrics(prediction.target.to_numpy(dtype=int), prediction.probability.to_numpy(dtype=float))
    payload = {**run_science, **metrics, "run_id": run_id, "inner_folds": 3, "selected_epochs": epochs, "pretraining": pretraining, "residual_alpha_mean": float(np.mean(alphas)), "parameter_count": PARAMETER_COUNT, "outer_labels_used": False, "prediction_path": prediction_path.relative_to(ROOT).as_posix()}
    prediction.to_parquet(prediction_path, index=False); write_json(result_path, payload)
    return payload


def _aggregate(rows: list[dict[str, Any]], candidate: str) -> dict[str, Any]:
    fields = ("accuracy", "balanced_accuracy", "macro_precision", "macro_recall", "macro_f1", "pr_auc", "roc_auc", "nll", "brier", "ece", "risk_precision", "risk_recall", "risk_f1")
    selected = [row for row in rows if row["candidate"] == candidate]
    result = {"candidate": candidate, "runs": len(selected)}
    for field in fields:
        result[field] = float(np.mean([float(row[field]) for row in selected]))
        result[f"std_{field}"] = float(np.std([float(row[field]) for row in selected]))
    return result


def _component_registry() -> dict[str, Any]:
    payload = {
        "R0": {"name": "Phase7 recipe", "valid": True},
        "R1": {"name": "H0 endpoint feature schema", "valid_subset": "score-free compact 49 mapped into frozen H1 165-input contract", "score_features": "REJECTED"},
        "R2": {"name": "H0 sequence preprocessing", "valid": True, "policy": "masked train-only mean/std"},
        "R3": {"name": "H0 pretraining", "valid_subset": list(LEGAL_PRETRAIN_TASKS), "rejected": list(REJECTED_PRETRAIN_TASKS)},
        "R4": {"name": "H0 loss/auxiliary", "valid": True, "weights": {"survival": 0.15, "outcome": 0.15}},
        "R5": {"name": "H0 epoch/checkpoint", "valid": True, "fixed_epochs": 8},
        "R6": {"name": "threshold", "valid": True, "policy": "pooled inner OOF Macro-F1"},
        "architecture": architecture_identity(),
    }
    write_json(OUT / "recovery_component_registry.json", payload)
    return payload


def _skipped_optuna_db() -> None:
    path = OUT / "stage_c_optuna.db"
    with sqlite3.connect(path) as connection:
        connection.execute("CREATE TABLE IF NOT EXISTS phase9_metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL)")
        connection.execute("INSERT OR REPLACE INTO phase9_metadata VALUES ('status', 'NOT_TRIGGERED')")


def run_supervisor() -> int:
    _directories(); started = utc_now(); before = early_warning_checksums()
    _status(
        state="RUNNING",
        started_at=started,
        finished_at=None,
        current_stage="audit",
        completed_runs=0,
        failed_runs=0,
        exit_code=None,
        pid=os.getpid(),
    )
    _sentinel("RUNNING")
    try:
        phase8_gate = json.loads(PHASE8_GATE.read_text(encoding="utf-8"))
        if phase8_gate.get("gate", phase8_gate.get("status")) != "PASS":
            raise RuntimeError("Phase 8 gate is not PASS")
        if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
            raise RuntimeError("Phase 9 requires exactly one CUDA GPU")
        identity = architecture_identity(); score = score_feature_authority(); holdout = holdout_availability_audit(); _component_registry()
        write_json(OUT / "early_warning_checksums_before.json", before)
        bundle = oulad._build_bundle()
        # Stage A: exact bounded control/recovery comparison.
        _status(current_stage="stage_a", current_candidate="A0_PHASE7_H1_CONTROL")
        stage_a: list[dict[str, Any]] = []
        for candidate, recipe in (("A0_PHASE7_H1_CONTROL", "A0_PHASE7_H1_CONTROL"), ("A1_VALID_H0_RECIPE", "A1_VALID_H0_RECIPE")):
            for outer in OUTER_FOLDS:
                config = control_config(outer) if candidate.startswith("A0") else copy.deepcopy(H0_CONFIG)
                result = evaluate_candidate(bundle, candidate=candidate, recipe=recipe, outer_fold=outer, seed=SEARCH_SEED, config=config)
                stage_a.append(result); _status(completed_runs=len(stage_a), current_candidate=candidate)
                _write_csv(OUT / "stage_a_reconstruction.csv", stage_a)
        a0 = _aggregate(stage_a, "A0_PHASE7_H1_CONTROL"); a1 = _aggregate(stage_a, "A1_VALID_H0_RECIPE")
        delta = {field: a1[field] - a0[field] for field in ("macro_f1", "pr_auc", "roc_auc", "nll", "brier", "ece")}
        material = delta["macro_f1"] >= 0.010 or delta["pr_auc"] >= 0.010 or delta["roc_auc"] >= 0.010
        write_json(OUT / "stage_a_summary.json", {"A0": a0, "A1": a1, "delta_A1_minus_A0": delta, "recipe_recovery_supported": material})
        # Stage B: bounded drop-one only when the preregistered gate passes.
        stage_b: list[dict[str, Any]] = []
        if material:
            _status(current_stage="stage_b")
            ablations = (
                ("B1_WITHOUT_SCORE_FEATURES", "NOT_APPLICABLE_SCORE_PROXY_REJECTED"),
                ("B2_WITHOUT_H0_PREPROCESSING", "A0_PHASE7_H1_CONTROL"),
                ("B3_WITHOUT_PRETRAINING", "B3_WITHOUT_PRETRAINING"),
                ("B4_WITHOUT_H0_TRAINING_POLICY", "B4_WITHOUT_H0_TRAINING_POLICY"),
            )
            for candidate, recipe in ablations:
                if candidate.startswith("B1"):
                    stage_b.append({"candidate": candidate, "status": "NOT_APPLICABLE", "reason": SCORE_DECISION}); continue
                for outer in OUTER_FOLDS:
                    config = control_config(outer) if candidate in {"B2_WITHOUT_H0_PREPROCESSING", "B4_WITHOUT_H0_TRAINING_POLICY"} else copy.deepcopy(H0_CONFIG)
                    actual_recipe = {
                        "B2_WITHOUT_H0_PREPROCESSING": "B2_WITHOUT_H0_PREPROCESSING",
                        "B3_WITHOUT_PRETRAINING": "B3_WITHOUT_PRETRAINING",
                        "B4_WITHOUT_H0_TRAINING_POLICY": "B4_WITHOUT_H0_TRAINING_POLICY",
                    }[candidate]
                    result = evaluate_candidate(bundle, candidate=candidate, recipe=actual_recipe, outer_fold=outer, seed=SEARCH_SEED, config=config)
                    stage_b.append(result); _status(completed_runs=len(stage_a) + len([x for x in stage_b if "macro_f1" in x]), current_candidate=candidate)
                    _write_csv(OUT / "stage_b_attribution.csv", stage_b)
        else:
            stage_b = [{"status": "NOT_TRIGGERED", "reason": "Stage A materiality gate failed"}]
        _write_csv(OUT / "stage_b_attribution.csv", stage_b)
        # Training tune is permitted only with explicit sensitivity evidence.
        policy_sensitive = False
        if material:
            full = a1["macro_f1"]
            b4_rows = [row for row in stage_b if row.get("candidate") == "B4_WITHOUT_H0_TRAINING_POLICY"]
            if b4_rows:
                policy_sensitive = full - float(np.mean([row["macro_f1"] for row in b4_rows])) >= 0.002
        tune_trigger = bool(material and a1["macro_f1"] < 0.83 and policy_sensitive)
        selected_config_by_fold = {fold: copy.deepcopy(H0_CONFIG) for fold in OUTER_FOLDS}
        stage_c_rows: list[dict[str, Any]] = []
        if tune_trigger:
            _status(current_stage="stage_c", optuna_triggered=True)
            study = optuna.create_study(study_name="phase9_h1r_training", storage=f"sqlite:///{(OUT / 'stage_c_optuna.db').as_posix()}", direction="maximize", sampler=optuna.samplers.TPESampler(seed=202609, n_startup_trials=6), pruner=optuna.pruners.MedianPruner(n_startup_trials=6, n_warmup_steps=3), load_if_exists=True)
            def objective(trial: optuna.Trial) -> float:
                outer = trial.number % 3
                config = {**H0_CONFIG, "learning_rate": trial.suggest_float("learning_rate", 2e-4, 2e-3, log=True), "weight_decay": trial.suggest_float("weight_decay", 1e-8, 5e-4, log=True), "dropout": trial.suggest_float("dropout", 0.15, 0.35), "batch_size": trial.suggest_categorical("batch_size", [128, 256]), "survival_weight": trial.suggest_categorical("survival_weight", [0.0, 0.10, 0.15, 0.20]), "outcome_weight": trial.suggest_categorical("outcome_weight", [0.0, 0.10, 0.15, 0.20])}
                result = evaluate_candidate(bundle, candidate="H1_R2_TUNED_VALID_H0_RECIPE", recipe="A1_VALID_H0_RECIPE", outer_fold=outer, seed=SEARCH_SEED, config=config)
                trial.set_user_attr("outer_fold", outer); trial.set_user_attr("config", config); trial.set_user_attr("metrics", {key: result[key] for key in ("macro_f1", "pr_auc", "roc_auc", "nll", "brier", "ece")}); trial.set_user_attr("outer_labels_used", False)
                return float(result["macro_f1"])
            remaining = max(0, MAX_OPTUNA_TRIALS - len(study.trials))
            if remaining:
                study.optimize(objective, n_trials=remaining, catch=(RuntimeError, FloatingPointError), gc_after_trial=True)
            for trial in study.trials:
                stage_c_rows.append({"number": trial.number, "state": trial.state.name, "value": trial.value, "outer_fold": trial.user_attrs.get("outer_fold"), "config": trial.user_attrs.get("config"), "metrics": trial.user_attrs.get("metrics"), "outer_labels_used": trial.user_attrs.get("outer_labels_used", False)})
            complete = [trial for trial in study.trials if trial.state == optuna.trial.TrialState.COMPLETE]
            for outer in OUTER_FOLDS:
                eligible = [trial for trial in complete if trial.user_attrs.get("outer_fold") == outer]
                if eligible:
                    best = max(eligible, key=lambda trial: (float(trial.value), float(trial.user_attrs["metrics"]["pr_auc"]), -float(trial.user_attrs["metrics"]["nll"])))
                    selected_config_by_fold[outer] = best.user_attrs["config"]
        else:
            _skipped_optuna_db(); stage_c_rows = [{"status": "NOT_TRIGGERED", "reason": "No material training-sensitivity trigger", "trials_scheduled": 0}]
        _write_csv(OUT / "stage_c_trials.csv", stage_c_rows)
        # Select A1 only when it improves; tuned R2 only when triggered.
        selected_name = "H1_R2_TUNED_VALID_H0_RECIPE" if tune_trigger else ("H1_R1_VALID_H0_RECIPE" if a1["macro_f1"] > a0["macro_f1"] else "H1_R0_PHASE7_CONTROL")
        selected_recipe = "A1_VALID_H0_RECIPE" if selected_name != "H1_R0_PHASE7_CONTROL" else "A0_PHASE7_H1_CONTROL"
        selected_frozen_configs = (
            {fold: control_config(fold) for fold in OUTER_FOLDS}
            if selected_name == "H1_R0_PHASE7_CONTROL"
            else selected_config_by_fold
        )
        # Stability uses exactly two predefined seeds across all three partitions; compare A0 and finalist.
        _status(current_stage="stability", current_candidate=selected_name)
        stability: list[dict[str, Any]] = []
        stability_finalist = (
            selected_name
            if selected_name != "H1_R0_PHASE7_CONTROL"
            else "H1_R1_VALID_H0_RECIPE"
        )
        stability_finalist_recipe = (
            selected_recipe
            if selected_name != "H1_R0_PHASE7_CONTROL"
            else "A1_VALID_H0_RECIPE"
        )
        for candidate, recipe in (
            ("H1_R0_PHASE7_CONTROL", "A0_PHASE7_H1_CONTROL"),
            (stability_finalist, stability_finalist_recipe),
        ):
            for outer in OUTER_FOLDS:
                config = (
                    control_config(outer)
                    if candidate == "H1_R0_PHASE7_CONTROL"
                    else selected_config_by_fold[outer]
                )
                for seed in STABILITY_SEEDS:
                    result = evaluate_candidate(bundle, candidate=candidate, recipe=recipe, outer_fold=outer, seed=seed, config=config)
                    stability.append(result); _status(completed_runs=len(stage_a) + len([x for x in stage_b if "macro_f1" in x]) + len(stability), current_candidate=candidate)
                    _write_csv(OUT / "stability.csv", stability)
        stability_summary = {candidate: _aggregate(stability, candidate) for candidate in sorted({row["candidate"] for row in stability})}
        # Historical H0/MLP development authority is retained as comparator evidence, not outer selection.
        h0_profile = json.loads(PHASE8_H0.read_text(encoding="utf-8")); h1_profile = json.loads(PHASE8_H1.read_text(encoding="utf-8"))
        write_json(OUT / "stability_summary.json", {"models": stability_summary, "historical_h0_endpoint_context": h0_profile["reproduced_metrics"], "historical_mlp_prediction_source": HISTORICAL_MLP_PRED.relative_to(ROOT).as_posix(), "phase7_outer_not_used_for_selection": True})
        # Contribution ablations are INNER-only and use the frozen selected recipe.
        _status(current_stage="hybrid_ablation")
        ablation_rows: list[dict[str, Any]] = []
        for candidate, disable_residual, disable_temporal in (("H1R_FULL", False, False), ("H1R_RESIDUAL_DISABLED", True, False), ("H1R_TEMPORAL_DISABLED", False, True)):
            for outer in OUTER_FOLDS:
                config = control_config(outer) if selected_name == "H1_R0_PHASE7_CONTROL" else selected_config_by_fold[outer]
                result = evaluate_candidate(bundle, candidate=candidate, recipe=selected_recipe, outer_fold=outer, seed=SEARCH_SEED, config=config, disable_residual=disable_residual, disable_temporal=disable_temporal)
                ablation_rows.append(result)
        residual_rows = [row for row in ablation_rows if row["candidate"] in {"H1R_FULL", "H1R_RESIDUAL_DISABLED"}]
        temporal_rows = [row for row in ablation_rows if row["candidate"] in {"H1R_FULL", "H1R_TEMPORAL_DISABLED"}]
        _write_csv(OUT / "residual_ablation.csv", residual_rows); _write_csv(OUT / "temporal_ablation.csv", temporal_rows)
        full = _aggregate(ablation_rows, "H1R_FULL"); residual_off = _aggregate(ablation_rows, "H1R_RESIDUAL_DISABLED"); temporal_off = _aggregate(ablation_rows, "H1R_TEMPORAL_DISABLED")
        finalist = stability_summary[selected_name]
        recovery_class = "STRONG_RECOVERY" if finalist["macro_f1"] >= 0.83 else "SUBSTANTIAL_RECOVERY" if finalist["macro_f1"] >= 0.82 else "PARTIAL_RECOVERY" if finalist["macro_f1"] >= 0.81 else "FAILED_RECOVERY"
        selected = {
            "candidate_id": selected_name, "architecture_id": ARCHITECTURE_ID, "architecture_hash": identity["architecture_hash"], "temporal_backbone_hash": identity["temporal_backbone_hash"], "parameter_count": PARAMETER_COUNT,
            "recipe": selected_recipe, "configs_by_outer_fold": selected_frozen_configs, "score_proxy_decision": SCORE_DECISION,
            "stability": finalist, "recovery_classification": recovery_class, "outer_labels_used": False,
            "true_untouched_holdout_available": False, "claim_scope": "RECOVERY_DEVELOPMENT_EVIDENCE",
        }
        write_json(OUT / "selected_candidate.json", selected)
        freeze = {**selected, "schema_version": "phase9_h1r_endpoint_development_freeze_v1", "feature_schema": "H0 compact score-free 49 mapped into frozen H1 165-dimensional contract" if selected_recipe != "A0_PHASE7_H1_CONTROL" else "Phase7 165-dimensional stage-safe schema", "preprocessing": "masked train-only sequence mean/std plus train-only aggregate/static transforms" if selected_recipe != "A0_PHASE7_H1_CONTROL" else "Phase7 train-only transforms", "pretraining": list(LEGAL_PRETRAIN_TASKS) if selected_recipe != "A0_PHASE7_H1_CONTROL" else [], "inner_folds": 3, "stability_seeds": list(STABILITY_SEEDS), "threshold_policy": "pooled inner OOF Macro-F1", "source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(), "frozen_at": utc_now(), "confirmation_opened": False, "freeze_hash": stable_hash(selected)}
        write_json(OUT / "H1_R_ENDPOINT_FREEZE_MANIFEST.json", freeze)
        write_json(OUT / "confirmation_metrics.json", {"status": "NOT_EXECUTED_NO_TRUE_UNTOUCHED_HOLDOUT", "outer_evaluations": 0, "labels_accessed": False})
        after = early_warning_checksums(); write_json(OUT / "early_warning_checksums_after.json", after)
        no_change = before == after
        gate = {
            "status": "PASS" if no_change else "FAIL", "score_authority_resolved": score["decision"], "early_warning_unchanged": no_change,
            "architecture_hash_count": 1, "parameter_count_count": 1, "stage_a_complete": len(stage_a) == 6,
            "candidate_selection_inner_only": True, "phase7_outer_used_for_tuning": False,
            "optuna_trials": len([row for row in stage_c_rows if "number" in row]), "optuna_within_budget": len([row for row in stage_c_rows if "number" in row]) <= MAX_OPTUNA_TRIALS,
            "architecture_search": False, "stability_complete": len(stability) == 12, "true_untouched_holdout_available": False,
            "confirmation_executed": False, "post_confirmation_tuning": False, "recovery_classification": recovery_class,
        }
        write_json(OUT / "phase9_gate.json", gate); write_json(OUT / "failure_summary.json", {"status": "NO_FAILURES", "failed_runs": 0})
        _status(state="COMPLETE", finished_at=utc_now(), current_stage="complete", current_candidate=selected_name, exit_code=0)
        _sentinel("COMPLETE", selected=selected_name, classification=recovery_class)
        return 0
    except Exception as error:
        write_json(OUT / "failure_summary.json", {"status": "FAILED", "type": type(error).__name__, "message": repr(error), "at": utc_now()})
        _status(state="FAILED", finished_at=utc_now(), exit_code=1, failed_runs=1)
        _sentinel("FAILED", error=repr(error))
        raise


__all__ = [
    "SCORE_DECISION", "architecture_identity", "early_warning_checksums",
    "evaluate_candidate", "holdout_availability_audit", "run_supervisor",
    "score_feature_authority", "valid_h0_aggregate",
]
