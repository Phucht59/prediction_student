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
from src.studies.v5_1.oulad.models import OULADHybridV51, count_parameters
from src.studies.v5_1.oulad.pretraining import load_pretrained_temporal
from src.studies.v5_1.oulad.runner import _inner_splits, _load
from src.studies.v5_1.oulad.training import classification_loss, deterministic_seed

from .contract import ARTIFACT_ROOT, REPORT_ROOT, ROOT, atomic_json, atomic_text, load_protocol
from .pretraining import _config


MULTITASK_ROOT = ARTIFACT_ROOT / "prediction/multitask"
HORIZON_WEEKS = 20


@dataclass(frozen=True)
class TemporalTargets:
    event_week: np.ndarray
    observation_week: np.ndarray
    withdrawal_event: np.ndarray
    outcome_target: np.ndarray
    withdrawal_day: np.ndarray


@dataclass(frozen=True)
class MultiTaskFit:
    binary_probability: np.ndarray
    hazard_probability: np.ndarray
    outcome_probability: np.ndarray
    state_dict: dict[str, torch.Tensor]
    checkpoint_sha256: str
    replay_max_abs_difference: float
    parameter_count: int
    runtime_seconds: float
    history: list[dict[str, float | int]]


class V6TemporalMultiTask(nn.Module):
    def __init__(
        self,
        sequence_channels: int,
        aggregate_dim: int,
        static_dim: int,
        config: dict[str, Any],
    ):
        super().__init__()
        self.backbone = OULADHybridV51(
            sequence_channels, aggregate_dim, static_dim, config, "cnn_bilstm"
        )
        fusion = int(config.get("fusion_hidden", 64))
        self.survival_head = nn.Linear(fusion, HORIZON_WEEKS)
        self.outcome_head = nn.Linear(fusion, 3)

    def representation(
        self,
        sequence: torch.Tensor,
        lengths: torch.Tensor,
        mask: torch.Tensor,
        aggregate: torch.Tensor,
        static: torch.Tensor,
    ) -> torch.Tensor:
        temporal, _, _ = self.backbone.temporal(sequence, lengths, mask)
        temporal = self.backbone.temporal_projection(temporal)
        aggregate_embedding = self.backbone._drop_branch(self.backbone.aggregate(aggregate))
        static_embedding = self.backbone._drop_branch(self.backbone.static(static))
        if self.backbone.gates is None:
            return torch.cat([temporal, aggregate_embedding, static_embedding], dim=1)
        gate = self.backbone.gates(
            torch.cat([temporal, aggregate_embedding, static_embedding], dim=1)
        )
        return temporal + gate[:, 0:1] * aggregate_embedding + gate[:, 1:2] * static_embedding

    def forward(
        self,
        sequence: torch.Tensor,
        lengths: torch.Tensor,
        mask: torch.Tensor,
        aggregate: torch.Tensor,
        static: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        representation = self.representation(sequence, lengths, mask, aggregate, static)
        return {
            "binary_logit": self.backbone.head(representation).squeeze(1),
            "hazard_logit": self.survival_head(representation),
            "outcome_logit": self.outcome_head(representation),
            "student_state_embedding": representation,
        }


def build_temporal_targets(data) -> TemporalTargets:
    keys = ["code_module", "code_presentation", "id_student"]
    cohort = data.base.cohort.reset_index(drop=True)
    registration = pd.read_csv(ROOT / "data/raw/studentRegistration.csv")
    info = pd.read_csv(ROOT / "data/raw/studentInfo.csv")
    frame = cohort.merge(
        registration[keys + ["date_unregistration"]], on=keys, validate="one_to_one"
    ).merge(info[keys + ["final_result"]], on=keys, validate="one_to_one")
    if not np.array_equal(frame.record_id.astype(str), cohort.record_id.astype(str)):
        raise RuntimeError("Temporal target merge changed record order")
    withdrawn = frame.final_result.eq("Withdrawn")
    valid_day = frame.date_unregistration.notna() & frame.date_unregistration.ge(0)
    after_cutoff = valid_day & frame.date_unregistration.gt(frame.cutoff_day)
    event = withdrawn & after_cutoff
    raw_event_week = np.floor(
        (frame.date_unregistration.fillna(frame.module_presentation_length) - frame.cutoff_day) / 7
    ).clip(lower=0, upper=HORIZON_WEEKS - 1)
    observation = np.floor(
        (frame.module_presentation_length - frame.cutoff_day).clip(lower=0) / 7
    ).clip(lower=0, upper=HORIZON_WEEKS - 1)
    event_week = np.where(event, raw_event_week, observation).astype(np.int64)
    outcome_map = {"Fail": 0, "Pass": 1, "Distinction": 2}
    outcome = frame.final_result.map(outcome_map).fillna(-1).astype(np.int64).to_numpy()
    return TemporalTargets(
        event_week=event_week,
        observation_week=np.where(event, event_week, observation).astype(np.int64),
        withdrawal_event=event.astype(np.float32).to_numpy(),
        outcome_target=outcome,
        withdrawal_day=frame.date_unregistration.fillna(-1).to_numpy(dtype=np.float32),
    )


def discrete_hazard_loss(
    hazard_logit: torch.Tensor,
    event_week: torch.Tensor,
    observation_week: torch.Tensor,
    event: torch.Tensor,
) -> torch.Tensor:
    weeks = torch.arange(hazard_logit.shape[1], device=hazard_logit.device).unsqueeze(0)
    at_risk = weeks <= observation_week.unsqueeze(1)
    target = (weeks == event_week.unsqueeze(1)) & event.bool().unsqueeze(1)
    losses = nn.functional.binary_cross_entropy_with_logits(
        hazard_logit, target.float(), reduction="none"
    )
    return losses.masked_select(at_risk).mean()


def _loader(
    inputs: OULADInputsV51,
    targets: TemporalTargets,
    indices: np.ndarray,
    *,
    batch_size: int,
    shuffle: bool,
    seed: int,
) -> DataLoader:
    dataset = TensorDataset(
        torch.from_numpy(inputs.sequence),
        torch.from_numpy(inputs.lengths),
        torch.from_numpy(inputs.mask),
        torch.from_numpy(inputs.aggregate),
        torch.from_numpy(inputs.static),
        torch.from_numpy(inputs.target),
        torch.from_numpy(targets.event_week[indices]),
        torch.from_numpy(targets.observation_week[indices]),
        torch.from_numpy(targets.withdrawal_event[indices]),
        torch.from_numpy(targets.outcome_target[indices]),
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        generator=torch.Generator().manual_seed(seed),
        num_workers=0,
        drop_last=False,
    )


def _state_hash(state: dict[str, torch.Tensor]) -> str:
    buffer = io.BytesIO()
    torch.save(state, buffer)
    return hashlib.sha256(buffer.getvalue()).hexdigest()


def _predict(
    model: V6TemporalMultiTask,
    inputs: OULADInputsV51,
    targets: TemporalTargets,
    indices: np.ndarray,
    device: torch.device,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    binary: list[np.ndarray] = []
    hazard: list[np.ndarray] = []
    outcome: list[np.ndarray] = []
    with torch.no_grad():
        for batch in _loader(
            inputs, targets, indices, batch_size=batch_size, shuffle=False, seed=0
        ):
            sequence, lengths, mask, aggregate, static = [value.to(device) for value in batch[:5]]
            output = model(sequence, lengths, mask, aggregate, static)
            binary.append(torch.sigmoid(output["binary_logit"]).cpu().numpy())
            hazard.append(torch.sigmoid(output["hazard_logit"]).cpu().numpy())
            outcome.append(torch.softmax(output["outcome_logit"], dim=1).cpu().numpy())
    return np.concatenate(binary), np.concatenate(hazard), np.concatenate(outcome)


def fit_multitask(
    train: OULADInputsV51,
    evaluation: OULADInputsV51,
    targets: TemporalTargets,
    train_indices: np.ndarray,
    evaluation_indices: np.ndarray,
    *,
    config: dict[str, Any],
    weights: dict[str, float],
    initial_temporal_state: dict[str, torch.Tensor],
    seed: int = 42,
    epochs: int = 8,
    device_name: str = "cuda",
) -> MultiTaskFit:
    started = time.perf_counter()
    deterministic_seed(seed)
    device = torch.device(
        device_name if device_name != "cuda" or torch.cuda.is_available() else "cpu"
    )
    model = V6TemporalMultiTask(
        train.sequence.shape[2], train.aggregate.shape[1], train.static.shape[1], config
    ).to(device)
    load_pretrained_temporal(model.backbone, initial_temporal_state)
    parameter_count = count_parameters(model)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config.get("learning_rate", 1e-3)),
        weight_decay=float(config.get("weight_decay", 1e-5)),
    )
    history: list[dict[str, float | int]] = []
    best_state = copy.deepcopy(model.state_dict())
    best_loss = float("inf")
    batch_size = int(config.get("batch_size", 256))
    for epoch in range(1, epochs + 1):
        model.train()
        losses: list[float] = []
        for batch in _loader(
            train,
            targets,
            train_indices,
            batch_size=batch_size,
            shuffle=True,
            seed=seed + epoch,
        ):
            values = [value.to(device) for value in batch]
            sequence, lengths, mask, aggregate, static, binary_target = values[:6]
            event_week, observation_week, event, outcome_target = values[6:]
            optimizer.zero_grad(set_to_none=True)
            output = model(sequence, lengths, mask, aggregate, static)
            binary_loss = classification_loss(
                output["binary_logit"],
                binary_target,
                loss_name="standard_bce",
                positive_weight=1.0,
                focal_gamma=2.0,
            )
            survival_loss = discrete_hazard_loss(
                output["hazard_logit"], event_week, observation_week, event
            )
            valid_outcome = outcome_target >= 0
            outcome_loss = nn.functional.cross_entropy(
                output["outcome_logit"][valid_outcome], outcome_target[valid_outcome].long()
            )
            withdrawal_risk = 1.0 - torch.prod(
                1.0 - torch.sigmoid(output["hazard_logit"]), dim=1
            )
            fail_probability = torch.softmax(output["outcome_logit"], dim=1)[:, 0]
            union_risk = 1.0 - (1.0 - withdrawal_risk) * (1.0 - fail_probability)
            consistency = nn.functional.mse_loss(
                torch.sigmoid(output["binary_logit"]), union_risk
            )
            loss = (
                binary_loss
                + float(weights["survival"]) * survival_loss
                + float(weights["outcome"]) * outcome_loss
                + 0.05 * consistency
            )
            if not torch.isfinite(loss):
                raise RuntimeError("Non-finite V6 multi-task loss")
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), float(config.get("gradient_clip", 1.0)))
            optimizer.step()
            losses.append(float(loss.detach()))
        mean_loss = float(np.mean(losses))
        history.append({"epoch": epoch, "loss": mean_loss})
        if mean_loss < best_loss:
            best_loss = mean_loss
            best_state = copy.deepcopy(model.state_dict())
    model.load_state_dict(best_state)
    binary, hazard, outcome = _predict(
        model, evaluation, targets, evaluation_indices, device, batch_size
    )
    cpu_state = {name: value.detach().cpu().clone() for name, value in best_state.items()}
    replay = V6TemporalMultiTask(
        train.sequence.shape[2], train.aggregate.shape[1], train.static.shape[1], config
    ).to(device)
    replay.load_state_dict(cpu_state)
    replay_binary, _, _ = _predict(
        replay, evaluation, targets, evaluation_indices, device, batch_size
    )
    difference = float(np.max(np.abs(binary - replay_binary)))
    return MultiTaskFit(
        binary_probability=binary,
        hazard_probability=hazard,
        outcome_probability=outcome,
        state_dict=cpu_state,
        checkpoint_sha256=_state_hash(cpu_state),
        replay_max_abs_difference=difference,
        parameter_count=parameter_count,
        runtime_seconds=time.perf_counter() - started,
        history=history,
    )


def _concordance(risk: np.ndarray, time_week: np.ndarray, event: np.ndarray) -> float:
    concordant = comparable = 0.0
    for index in np.flatnonzero(event > 0):
        comparable_rows = time_week > time_week[index]
        if not comparable_rows.any():
            continue
        differences = risk[index] - risk[comparable_rows]
        concordant += float((differences > 0).sum()) + 0.5 * float((differences == 0).sum())
        comparable += float(comparable_rows.sum())
    return concordant / comparable if comparable else 0.5


def _candidate_metrics(frame: pd.DataFrame, hazard: np.ndarray, outcome: np.ndarray) -> dict[str, Any]:
    target = frame.target.to_numpy(dtype=int)
    probability = frame.probability.to_numpy(dtype=float)
    predicted = probability >= 0.5
    event = frame.withdrawal_event.to_numpy(dtype=int)
    time_week = frame.observation_week.to_numpy(dtype=int)
    cumulative = 1.0 - np.prod(1.0 - hazard, axis=1)
    valid_outcome = frame.outcome_target.to_numpy(dtype=int) >= 0
    outcome_target = frame.outcome_target.to_numpy(dtype=int)[valid_outcome]
    outcome_prediction = outcome[valid_outcome].argmax(axis=1)
    event_recall = float((cumulative[event > 0] >= 0.5).mean()) if event.any() else 0.0
    lead = frame.withdrawal_day.to_numpy(dtype=float) - frame.cutoff_day.to_numpy(dtype=float)
    predicted_event_lead = lead[(event > 0) & (cumulative >= 0.5)]
    observed = np.arange(HORIZON_WEEKS)[None, :] <= time_week[:, None]
    event_by_week = (
        (np.arange(HORIZON_WEEKS)[None, :] >= time_week[:, None]) & (event[:, None] > 0)
    ).astype(float)
    cumulative_by_week = 1.0 - np.cumprod(1.0 - hazard, axis=1)
    return {
        "macro_f1": float(f1_score(target, predicted, average="macro", zero_division=0)),
        "at_risk_f1": float(f1_score(target, predicted, zero_division=0)),
        "at_risk_recall": float(
            ((predicted & (target == 1)).sum() / max(1, (target == 1).sum()))
        ),
        "pr_auc": float(average_precision_score(target, probability)),
        "brier": float(brier_score_loss(target, probability)),
        "survival_concordance": float(_concordance(cumulative, time_week, event)),
        "integrated_brier_observed_risk_sets": float(
            ((cumulative_by_week - event_by_week) ** 2)[observed].mean()
        ),
        "withdrawal_recall": event_recall,
        "median_warning_lead_days": float(np.median(predicted_event_lead))
        if len(predicted_event_lead)
        else None,
        "outcome_macro_f1": float(
            f1_score(outcome_target, outcome_prediction, average="macro", zero_division=0)
        ),
        "outcome_majority_macro_f1": float(
            f1_score(
                outcome_target,
                np.full_like(outcome_target, np.bincount(outcome_target).argmax()),
                average="macro",
                zero_division=0,
            )
        ),
    }


def screen_multitask(device_name: str = "cuda") -> dict[str, Any]:
    output = MULTITASK_ROOT / "gate.json"
    if output.is_file():
        result = json.loads(output.read_text(encoding="utf-8"))
        if result.get("status") == "COMPLETE":
            return result
    pretraining_gate = json.loads(
        (ARTIFACT_ROOT / "prediction/pretraining/gate.json").read_text(encoding="utf-8")
    )
    audit = json.loads((ARTIFACT_ROOT / "audit/knowledge_audit.json").read_text(encoding="utf-8"))
    eligible = bool(
        pretraining_gate["gate_pass"]
        or audit["order_destruction"]["verdict"]
        in {"TEMPORAL_ORDER_HIGH_VALUE", "TEMPORAL_ORDER_MODERATE_VALUE"}
    )
    if not eligible:
        result = {
            "schema_version": "v6_multitask_gate_v1",
            "status": "COMPLETE",
            "selected": "B_MINIMAL_PRETRAINING",
            "gate_pass": False,
            "skip_reason": "SKIPPED_NO_TEMPORAL_SIGNAL_GATE",
            "outer_test_accessed": False,
            "future_accessed": False,
        }
        atomic_json(output, result)
        return result
    protocol = load_protocol()
    _, v4_protocol, data = _load()
    targets = build_temporal_targets(data)
    splits = _inner_splits(data, 0, v4_protocol)
    config = _config()
    weight_sets = list(protocol["multitask"]["registered_weights"])
    rows: list[dict[str, Any]] = []
    prediction_rows: list[dict[str, Any]] = []
    auxiliary: dict[str, list[np.ndarray]] = {f"W{index}": [] for index in range(len(weight_sets))}
    checkpoint_root = MULTITASK_ROOT / "checkpoints"
    checkpoint_root.mkdir(parents=True, exist_ok=True)
    for inner_fold, (train_index, validation_index) in enumerate(splits):
        train = prepare_oulad_inputs(data, train_index, train_index)
        validation = prepare_oulad_inputs(
            data, train_index, validation_index, fitted=train.preprocessors
        )
        temporal_state = torch.load(
            ARTIFACT_ROOT
            / f"prediction/pretraining/checkpoints/p1_inner_{inner_fold}_seed_42.pt",
            map_location="cpu",
            weights_only=True,
        )
        for weight_index, weights in enumerate(weight_sets):
            candidate = f"W{weight_index}"
            fit = fit_multitask(
                train,
                validation,
                targets,
                train_index,
                validation_index,
                config=config,
                weights=weights,
                initial_temporal_state=temporal_state,
                seed=42,
                epochs=8,
                device_name=device_name,
            )
            checkpoint = checkpoint_root / f"{candidate}_inner_{inner_fold}_seed_42.pt"
            torch.save(fit.state_dict, checkpoint)
            frame = pd.DataFrame(
                {
                    "record_id": data.base.record_ids[validation_index].astype(str),
                    "inner_fold": inner_fold,
                    "target": data.y[validation_index].astype(int),
                    "probability": fit.binary_probability,
                    "withdrawal_event": targets.withdrawal_event[validation_index],
                    "observation_week": targets.observation_week[validation_index],
                    "outcome_target": targets.outcome_target[validation_index],
                    "withdrawal_day": targets.withdrawal_day[validation_index],
                    "cutoff_day": data.base.cohort.iloc[validation_index].cutoff_day.to_numpy(),
                }
            )
            metrics = _candidate_metrics(frame, fit.hazard_probability, fit.outcome_probability)
            rows.append(
                {
                    "candidate": candidate,
                    "weights": weights,
                    "inner_fold": inner_fold,
                    **metrics,
                    "parameter_count": fit.parameter_count,
                    "runtime_seconds": fit.runtime_seconds,
                    "checkpoint_sha256": fit.checkpoint_sha256,
                    "replay_max_abs_difference": fit.replay_max_abs_difference,
                }
            )
            prediction_rows.extend(
                {
                    "candidate": candidate,
                    **row,
                }
                for row in frame.to_dict(orient="records")
            )
            auxiliary[candidate].append(
                np.column_stack(
                    [
                        fit.hazard_probability,
                        fit.outcome_probability,
                    ]
                )
            )
        pd.DataFrame(rows).to_json(MULTITASK_ROOT / "fold_metrics.json", orient="records", indent=2)
        pd.DataFrame(prediction_rows).to_parquet(MULTITASK_ROOT / "inner_oof.parquet", index=False)
    metrics = pd.DataFrame(rows)
    aggregate = metrics.groupby("candidate").mean(numeric_only=True)
    baseline = pretraining_gate["candidates"]["P1_MASKED_AND_NEXT_WEEK"]
    candidates = {}
    for candidate, row in aggregate.iterrows():
        binary_ok = bool(
            float(row.macro_f1) >= float(baseline["macro_f1"]) - 0.001
            and float(row.brier) <= float(baseline["brier"]) + 0.002
        )
        survival_ok = float(row.survival_concordance) >= 0.55
        outcome_ok = float(row.outcome_macro_f1) >= float(row.outcome_majority_macro_f1) + 0.05
        candidates[candidate] = {
            **row.to_dict(),
            "weights": weight_sets[int(candidate[1:])],
            "binary_noninferior": binary_ok,
            "survival_useful": survival_ok,
            "outcome_exceeds_majority": outcome_ok,
            "gate_pass": binary_ok and survival_ok and outcome_ok,
        }
    passing = [name for name, row in candidates.items() if row["gate_pass"]]
    selected = (
        max(passing, key=lambda name: (candidates[name]["macro_f1"], candidates[name]["pr_auc"]))
        if passing
        else None
    )
    result = {
        "schema_version": "v6_multitask_gate_v1",
        "status": "COMPLETE",
        "selected": selected if selected is not None else "B_MINIMAL_PRETRAINING",
        "gate_pass": selected is not None,
        "candidates": candidates,
        "baseline": baseline,
        "survival_event": "withdrawal_only",
        "fail_event_time_assumed": False,
        "outer_test_accessed": False,
        "future_accessed": False,
    }
    atomic_json(output, result)
    rows_text = "\n".join(
        f"| {name} | {row['macro_f1']:.6f} | {row['pr_auc']:.6f} | "
        f"{row['survival_concordance']:.6f} | {row['outcome_macro_f1']:.6f} | "
        f"{str(row['gate_pass']).lower()} |"
        for name, row in candidates.items()
    )
    atomic_text(
        REPORT_ROOT / "TEMPORAL_MULTITASK_REPORT.md",
        f"""# V6 temporal multi-task report

| Candidate | Macro-F1 | PR-AUC | Survival C-index | Outcome Macro-F1 | Gate |
|---|---:|---:|---:|---:|---:|
{rows_text}

Selected: **{result['selected']}**. Withdrawal is the only time-to-event target;
Fail is a masked final-outcome class and is never assigned a fabricated event
time. All screening is confined to outer-training fold 0.
""",
    )
    return result


__all__ = [
    "HORIZON_WEEKS",
    "TemporalTargets",
    "V6TemporalMultiTask",
    "build_temporal_targets",
    "discrete_hazard_loss",
    "fit_multitask",
    "screen_multitask",
]

