from __future__ import annotations

import copy
import hashlib
import io
import json
import random
import time
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score, brier_score_loss, f1_score
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from src.studies.v5_1.oulad.data import OULADInputsV51, prepare_oulad_inputs
from src.studies.v5_1.oulad.models import OULADTemporalEncoderV51, count_parameters
from src.studies.v5_1.oulad.runner import _inner_splits, _load
from src.studies.v5_1.oulad.training import fit_prepared_oulad_model

from .contract import ARTIFACT_ROOT, REPORT_ROOT, ROOT, atomic_json, atomic_text, load_protocol


PRETRAIN_ROOT = ARTIFACT_ROOT / "prediction/pretraining"
PRETRAINING_TASKS = (
    "masked_activity_band",
    "masked_active_state",
    "masked_submission_band",
    "masked_score_availability",
    "masked_inactivity_transition",
    "next_active_state",
    "next_activity_direction",
    "next_submission_band",
    "next_inactivity_transition",
    "next_score_state_transition",
)


@dataclass(frozen=True)
class MinimalPretrainingResult:
    temporal_state_dict: dict[str, torch.Tensor]
    checkpoint_sha256: str
    history: list[dict[str, float | int]]
    parameter_count: int
    runtime_seconds: float
    replay_max_abs_difference: float
    tasks: tuple[str, ...]


class MinimalTemporalPretrainer(nn.Module):
    """V5.1 encoder with only the registered masked- and next-week heads."""

    def __init__(self, input_channels: int, config: dict[str, Any]):
        super().__init__()
        self.encoder = OULADTemporalEncoderV51(input_channels, config, "cnn_bilstm")
        width = self.encoder.sequence_output_dim
        self.masked_activity_band = nn.Linear(width, 3)
        self.masked_active = nn.Linear(width, 1)
        self.masked_submission_band = nn.Linear(width, 3)
        self.masked_score_available = nn.Linear(width, 1)
        self.masked_inactivity_transition = nn.Linear(width, 1)
        self.next_active = nn.Linear(width, 1)
        self.next_activity_direction = nn.Linear(width, 3)
        self.next_submission_band = nn.Linear(width, 3)
        self.next_inactivity_transition = nn.Linear(width, 1)
        self.next_score_transition = nn.Linear(width, 3)

    def forward(
        self, sequence: torch.Tensor, lengths: torch.Tensor, mask: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        values = self.encoder.encode_sequence(sequence, lengths, mask)
        return {
            "masked_activity_band": self.masked_activity_band(values),
            "masked_active": self.masked_active(values).squeeze(-1),
            "masked_submission_band": self.masked_submission_band(values),
            "masked_score_available": self.masked_score_available(values).squeeze(-1),
            "masked_inactivity_transition": self.masked_inactivity_transition(values).squeeze(-1),
            "next_active": self.next_active(values).squeeze(-1),
            "next_activity_direction": self.next_activity_direction(values),
            "next_submission_band": self.next_submission_band(values),
            "next_inactivity_transition": self.next_inactivity_transition(values).squeeze(-1),
            "next_score_transition": self.next_score_transition(values),
        }


def _seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)


def _state_hash(state: dict[str, torch.Tensor]) -> str:
    buffer = io.BytesIO()
    torch.save(state, buffer)
    return hashlib.sha256(buffer.getvalue()).hexdigest()


def _channel_indices(order: tuple[str, ...]) -> dict[str, int]:
    required = [
        "total_clicks",
        "submitted_assessment_count",
        "available_score_count",
        "weeks_without_activity",
    ]
    missing = sorted(set(required) - set(order))
    if missing:
        raise ValueError(f"V6 pretraining channels missing: {missing}")
    return {name: order.index(name) for name in required}


def _labels(
    raw: np.ndarray, mask: np.ndarray, order: tuple[str, ...]
) -> dict[str, np.ndarray]:
    columns = _channel_indices(order)
    clicks = raw[:, :, columns["total_clicks"]]
    submissions = raw[:, :, columns["submitted_assessment_count"]]
    scores = raw[:, :, columns["available_score_count"]]
    inactivity = raw[:, :, columns["weeks_without_activity"]]
    positive_clicks = clicks[(mask > 0) & (clicks > 0)]
    lower, upper = (
        np.quantile(positive_clicks, [0.33, 0.66]) if len(positive_clicks) else (1.0, 2.0)
    )
    activity_band = np.where(clicks <= 0, 0, np.where(clicks <= lower, 1, 2)).astype(np.int64)
    submission_band = np.where(submissions <= 0, 0, np.where(submissions < 2, 1, 2)).astype(
        np.int64
    )
    active = (clicks > 0).astype(np.float32)
    score_available = (scores > 0).astype(np.float32)
    inactivity_transition = np.zeros_like(active)
    inactivity_transition[:, 1:] = (inactivity[:, 1:] > inactivity[:, :-1]).astype(np.float32)
    direction = np.ones_like(activity_band)
    direction[:, :-1] = np.where(
        clicks[:, 1:] < clicks[:, :-1],
        0,
        np.where(clicks[:, 1:] > clicks[:, :-1], 2, 1),
    )
    score_transition = np.ones_like(activity_band)
    score_transition[:, :-1] = np.where(
        scores[:, 1:] < scores[:, :-1],
        0,
        np.where(scores[:, 1:] > scores[:, :-1], 2, 1),
    )
    return {
        "activity_band": activity_band,
        "active": active,
        "submission_band": submission_band,
        "score_available": score_available,
        "inactivity_transition": inactivity_transition,
        "activity_direction": direction,
        "score_transition": score_transition,
    }


def _week_mask(valid: np.ndarray, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    selected = np.zeros_like(valid, dtype=bool)
    for row, row_valid in enumerate(valid.astype(bool)):
        positions = np.flatnonzero(row_valid)
        if len(positions) <= 1:
            continue
        count = min(2, max(1, len(positions) // 10))
        selected[row, rng.choice(positions, size=count, replace=False)] = True
    return selected


def _loss(
    output: dict[str, torch.Tensor],
    labels: dict[str, torch.Tensor],
    masked: torch.Tensor,
    next_valid: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, float]]:
    binary = nn.functional.binary_cross_entropy_with_logits
    masked_losses = [
        nn.functional.cross_entropy(output["masked_activity_band"][masked], labels["activity_band"][masked]),
        binary(output["masked_active"][masked], labels["active"][masked]),
        nn.functional.cross_entropy(
            output["masked_submission_band"][masked], labels["submission_band"][masked]
        ),
        binary(output["masked_score_available"][masked], labels["score_available"][masked]),
        binary(
            output["masked_inactivity_transition"][masked],
            labels["inactivity_transition"][masked],
        ),
    ]
    next_losses = [
        binary(output["next_active"][:, :-1][next_valid], labels["active"][:, 1:][next_valid]),
        nn.functional.cross_entropy(
            output["next_activity_direction"][:, :-1][next_valid],
            labels["activity_direction"][:, :-1][next_valid],
        ),
        nn.functional.cross_entropy(
            output["next_submission_band"][:, :-1][next_valid],
            labels["submission_band"][:, 1:][next_valid],
        ),
        binary(
            output["next_inactivity_transition"][:, :-1][next_valid],
            labels["inactivity_transition"][:, 1:][next_valid],
        ),
        nn.functional.cross_entropy(
            output["next_score_transition"][:, :-1][next_valid],
            labels["score_transition"][:, :-1][next_valid],
        ),
    ]
    masked_loss = torch.stack(masked_losses).mean()
    next_loss = torch.stack(next_losses).mean()
    return 0.5 * (masked_loss + next_loss), {
        "masked_loss": float(masked_loss.detach()),
        "next_week_loss": float(next_loss.detach()),
    }


def fit_minimal_pretraining(
    train: OULADInputsV51,
    raw_sequence: np.ndarray,
    *,
    dynamic_channel_order: tuple[str, ...],
    config: dict[str, Any],
    seed: int,
    epochs: int = 5,
    device_name: str = "cuda",
) -> MinimalPretrainingResult:
    started = time.perf_counter()
    _seed(seed)
    device = torch.device(
        device_name if device_name != "cuda" or torch.cuda.is_available() else "cpu"
    )
    model = MinimalTemporalPretrainer(train.sequence.shape[2], config).to(device)
    parameter_count = count_parameters(model)
    if parameter_count >= int(config.get("parameter_limit", 1_500_000)):
        raise RuntimeError("V6 minimal pretraining parameter limit exceeded")
    label_values = _labels(raw_sequence, train.mask, dynamic_channel_order)
    tensors = [
        torch.from_numpy(train.sequence),
        torch.from_numpy(train.lengths),
        torch.from_numpy(train.mask),
        *(torch.from_numpy(label_values[name]) for name in label_values),
    ]
    dataset = TensorDataset(*tensors)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config.get("pretraining_learning_rate", 5e-4)),
        weight_decay=float(config.get("weight_decay", 1e-5)),
    )
    history: list[dict[str, float | int]] = []
    best_loss = float("inf")
    best_state = copy.deepcopy(model.state_dict())
    label_names = list(label_values)
    for epoch in range(1, epochs + 1):
        loader = DataLoader(
            dataset,
            batch_size=int(config.get("batch_size", 256)),
            shuffle=True,
            generator=torch.Generator().manual_seed(seed + epoch),
            num_workers=0,
            drop_last=False,
        )
        model.train()
        total: list[float] = []
        masked_parts: list[float] = []
        next_parts: list[float] = []
        offset = 0
        for batch in loader:
            sequence, lengths, valid = batch[:3]
            labels = {name: batch[index + 3].to(device) for index, name in enumerate(label_names)}
            selected = _week_mask(valid.numpy(), seed + epoch * 1_000_003 + offset)
            offset += len(sequence)
            corrupted = sequence.clone()
            corrupted[torch.from_numpy(selected)] = 0.0
            optimizer.zero_grad(set_to_none=True)
            output = model(corrupted.to(device), lengths.to(device), valid.to(device))
            masked = torch.from_numpy(selected).to(device)
            next_valid = (valid[:, :-1].bool() & valid[:, 1:].bool()).to(device)
            loss, parts = _loss(output, labels, masked, next_valid)
            if not torch.isfinite(loss):
                raise RuntimeError("Non-finite V6 pretraining loss")
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), float(config.get("gradient_clip", 1.0)))
            optimizer.step()
            total.append(float(loss.detach()))
            masked_parts.append(parts["masked_loss"])
            next_parts.append(parts["next_week_loss"])
        mean_loss = float(np.mean(total))
        history.append(
            {
                "epoch": epoch,
                "loss": mean_loss,
                "masked_loss": float(np.mean(masked_parts)),
                "next_week_loss": float(np.mean(next_parts)),
            }
        )
        if mean_loss < best_loss:
            best_loss = mean_loss
            best_state = copy.deepcopy(model.state_dict())
    model.load_state_dict(best_state)
    cpu_state = {name: value.detach().cpu().clone() for name, value in best_state.items()}
    temporal = {
        f"temporal.{name[len('encoder.'):]}": value
        for name, value in cpu_state.items()
        if name.startswith("encoder.")
    }
    replay = MinimalTemporalPretrainer(train.sequence.shape[2], config).to(device)
    replay.load_state_dict(cpu_state)
    sample = torch.from_numpy(train.sequence[:8]).to(device)
    lengths = torch.from_numpy(train.lengths[:8]).to(device)
    mask = torch.from_numpy(train.mask[:8]).to(device)
    model.eval()
    replay.eval()
    with torch.no_grad():
        first = model(sample, lengths, mask)["next_active"]
        second = replay(sample, lengths, mask)["next_active"]
    difference = float((first - second).abs().max().cpu())
    return MinimalPretrainingResult(
        temporal_state_dict=temporal,
        checkpoint_sha256=_state_hash(cpu_state),
        history=history,
        parameter_count=parameter_count,
        runtime_seconds=time.perf_counter() - started,
        replay_max_abs_difference=difference,
        tasks=PRETRAINING_TASKS,
    )


def _config() -> dict[str, Any]:
    config = json.loads(
        (ROOT / "artifacts/v5_1/oulad/selected_configs.json").read_text(encoding="utf-8")
    )[0]["config"]
    return {
        **config,
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
    }


def _finalize_pretraining_gate(metrics: pd.DataFrame, protocol: dict[str, Any]) -> dict[str, Any]:
    aggregate = metrics.groupby("candidate").agg(
        macro_f1=("macro_f1", "mean"),
        at_risk_f1=("at_risk_f1", "mean"),
        pr_auc=("pr_auc", "mean"),
        brier=("brier", "mean"),
    )
    p0 = aggregate.loc["P0_NONE"]
    p1 = aggregate.loc["P1_MASKED_AND_NEXT_WEEK"]
    macro_wide = metrics.pivot(index="inner_fold", columns="candidate", values="macro_f1")
    pr_wide = metrics.pivot(index="inner_fold", columns="candidate", values="pr_auc")
    positive_macro = int((macro_wide["P1_MASKED_AND_NEXT_WEEK"] > macro_wide["P0_NONE"]).sum())
    positive_pr = int((pr_wide["P1_MASKED_AND_NEXT_WEEK"] > pr_wide["P0_NONE"]).sum())
    gain_macro = float(p1.macro_f1 - p0.macro_f1)
    gain_pr = float(p1.pr_auc - p0.pr_auc)
    gate = protocol["pretraining"]["gate"]
    minimum_gain = float(gate["macro_f1_or_pr_auc_gain"])
    minimum_positive = int(gate["positive_inner_folds"])
    macro_qualifies = gain_macro >= minimum_gain and positive_macro >= minimum_positive
    pr_qualifies = gain_pr >= minimum_gain and positive_pr >= minimum_positive
    basis = "macro_f1" if macro_qualifies else "pr_auc" if pr_qualifies else "none"
    positive = positive_macro if basis == "macro_f1" else positive_pr if basis == "pr_auc" else 0
    passed = bool(
        (macro_qualifies or pr_qualifies)
        and float(p1.at_risk_f1 - p0.at_risk_f1) >= -float(gate["at_risk_f1_max_drop"])
        and float(p1.brier - p0.brier) <= float(gate["brier_max_increase"])
    )
    result = {
        "schema_version": "v6_pretraining_gate_v1",
        "status": "COMPLETE",
        "selected": "P1_MASKED_AND_NEXT_WEEK" if passed else "P0_NONE",
        "gate_pass": passed,
        "gate_logic": "qualifying_metric_positive_folds_v1",
        "qualifying_metric": basis,
        "scope": "outer_training_fold_0_three_inner_folds_seed_42",
        "fixed_finetune_epochs": 8,
        "pretraining_epochs": 5,
        "candidates": {name: row.to_dict() for name, row in aggregate.iterrows()},
        "macro_f1_gain": gain_macro,
        "pr_auc_gain": gain_pr,
        "at_risk_f1_gain": float(p1.at_risk_f1 - p0.at_risk_f1),
        "brier_change": float(p1.brier - p0.brier),
        "positive_inner_folds": positive,
        "positive_macro_f1_inner_folds": positive_macro,
        "positive_pr_auc_inner_folds": positive_pr,
        "tasks": list(PRETRAINING_TASKS),
        "outer_test_accessed": False,
        "future_accessed": False,
    }
    atomic_json(PRETRAIN_ROOT / "gate.json", result)
    atomic_text(
        REPORT_ROOT / "PRETRAINING_REPORT.md",
        f"""# V6 minimal temporal pretraining

Status: **{'GATE_PASS' if passed else 'GATE_FAIL'}**  
Selected: **{result['selected']}**

P1 adds only masked-week categorical/state reconstruction and next-week state
prediction to the locked V5.1 temporal encoder. It uses outer-training fold 0,
three inner folds and seed 42; outer test and Future OULAD are not accessed.

- Mean Macro-F1 gain: {gain_macro:+.6f}
- Mean PR-AUC gain: {gain_pr:+.6f}
- Mean At-risk F1 gain: {float(p1.at_risk_f1 - p0.at_risk_f1):+.6f}
- Mean Brier change: {float(p1.brier - p0.brier):+.6f}
- Positive Macro-F1 inner folds: {positive_macro}/3
- Positive PR-AUC inner folds: {positive_pr}/3
- Qualifying metric: {basis}

The gate requires Macro-F1 or PR-AUC gain of at least `0.002`, at least two
positive folds for that qualifying metric, and the registered At-risk F1/Brier
guardrails.
""",
    )
    return result


def screen_pretraining(device_name: str = "cuda") -> dict[str, Any]:
    output = PRETRAIN_ROOT / "gate.json"
    if output.is_file():
        result = json.loads(output.read_text(encoding="utf-8"))
        if result.get("status") == "COMPLETE" and result.get("gate_logic") == "qualifying_metric_positive_folds_v1":
            return result
        metrics_path = PRETRAIN_ROOT / "fold_metrics.json"
        if metrics_path.is_file():
            cached = pd.read_json(metrics_path)
            if len(cached) == 6:
                return _finalize_pretraining_gate(cached, load_protocol())
    protocol = load_protocol()
    _, v4_protocol, data = _load()
    splits = _inner_splits(data, 0, v4_protocol)
    config = _config()
    rows: list[dict[str, Any]] = []
    predictions: list[dict[str, Any]] = []
    checkpoint_root = PRETRAIN_ROOT / "checkpoints"
    checkpoint_root.mkdir(parents=True, exist_ok=True)
    for inner_fold, (train_index, validation_index) in enumerate(splits):
        train = prepare_oulad_inputs(data, train_index, train_index)
        validation = prepare_oulad_inputs(
            data, train_index, validation_index, fitted=train.preprocessors
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
        checkpoint = checkpoint_root / f"p1_inner_{inner_fold}_seed_42.pt"
        torch.save(pretrain.temporal_state_dict, checkpoint)
        for candidate, initial_state in [
            ("P0_NONE", None),
            ("P1_MASKED_AND_NEXT_WEEK", pretrain.temporal_state_dict),
        ]:
            fit = fit_prepared_oulad_model(
                train,
                validation,
                config=config,
                seed=42,
                fixed_epochs=8,
                device_name=device_name,
                initial_temporal_state=initial_state,
            )
            predicted = fit.probability >= 0.5
            target = validation.target.astype(int)
            rows.append(
                {
                    "candidate": candidate,
                    "inner_fold": inner_fold,
                    "macro_f1": float(f1_score(target, predicted, average="macro", zero_division=0)),
                    "at_risk_f1": float(f1_score(target, predicted, zero_division=0)),
                    "pr_auc": float(average_precision_score(target, fit.probability)),
                    "brier": float(brier_score_loss(target, fit.probability)),
                    "parameter_count": fit.parameter_count,
                    "runtime_seconds": fit.runtime_seconds,
                    "pretraining_runtime_seconds": pretrain.runtime_seconds
                    if candidate == "P1_MASKED_AND_NEXT_WEEK"
                    else 0.0,
                    "pretraining_checkpoint_sha256": pretrain.checkpoint_sha256
                    if candidate == "P1_MASKED_AND_NEXT_WEEK"
                    else None,
                    "pretraining_replay_max_abs_difference": pretrain.replay_max_abs_difference
                    if candidate == "P1_MASKED_AND_NEXT_WEEK"
                    else None,
                }
            )
            predictions.extend(
                {
                    "candidate": candidate,
                    "inner_fold": inner_fold,
                    "record_id": str(data.base.record_ids[index]),
                    "target": int(data.y[index]),
                    "probability": float(probability),
                }
                for index, probability in zip(validation_index, fit.probability)
            )
        pd.DataFrame(rows).to_json(PRETRAIN_ROOT / "fold_metrics.json", orient="records", indent=2)
        pd.DataFrame(predictions).to_parquet(PRETRAIN_ROOT / "inner_oof.parquet", index=False)
    return _finalize_pretraining_gate(pd.DataFrame(rows), protocol)


__all__ = [
    "MinimalPretrainingResult",
    "MinimalTemporalPretrainer",
    "fit_minimal_pretraining",
    "screen_pretraining",
]
