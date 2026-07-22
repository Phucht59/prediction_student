from __future__ import annotations

import copy
import hashlib
import io
import json
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
from src.studies.v5_1.oulad.models import count_parameters
from src.studies.v5_1.oulad.runner import _inner_splits, _load
from src.studies.v5_1.oulad.training import classification_loss, deterministic_seed

from .contract import ARTIFACT_ROOT, REPORT_ROOT, atomic_json, atomic_text, load_protocol
from .multitask import (
    TemporalTargets,
    V6TemporalMultiTask,
    _candidate_metrics,
    _loader,
    build_temporal_targets,
    discrete_hazard_loss,
)
from .pretraining import _config


RANKING_ROOT = ARTIFACT_ROOT / "prediction/ranking"


class V6RiskRanking(V6TemporalMultiTask):
    def __init__(
        self,
        sequence_channels: int,
        aggregate_dim: int,
        static_dim: int,
        config: dict[str, Any],
    ):
        super().__init__(sequence_channels, aggregate_dim, static_dim, config)
        self.ranking_head = nn.Linear(int(config.get("fusion_hidden", 64)), 1)

    def forward(
        self,
        sequence: torch.Tensor,
        lengths: torch.Tensor,
        mask: torch.Tensor,
        aggregate: torch.Tensor,
        static: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        output = super().forward(sequence, lengths, mask, aggregate, static)
        output["ranking_score"] = self.ranking_head(
            output["student_state_embedding"]
        ).squeeze(1)
        return output


@dataclass(frozen=True)
class RankingFit:
    binary_probability: np.ndarray
    hazard_probability: np.ndarray
    outcome_probability: np.ndarray
    ranking_score: np.ndarray
    state_dict: dict[str, torch.Tensor]
    checkpoint_sha256: str
    replay_max_abs_difference: float
    parameter_count: int
    runtime_seconds: float
    pair_count: int


def _state_hash(state: dict[str, torch.Tensor]) -> str:
    buffer = io.BytesIO()
    torch.save(state, buffer)
    return hashlib.sha256(buffer.getvalue()).hexdigest()


def _pair_indices(
    cohort: pd.DataFrame,
    target: np.ndarray,
    teacher_probability: np.ndarray,
) -> np.ndarray:
    frame = cohort.reset_index(drop=True).copy()
    frame["target"] = target.astype(int)
    frame["teacher_probability"] = teacher_probability
    frame["progress_bucket"] = (
        frame.valid_sequence_length.astype(float).div(2).round().astype(int)
    )
    pairs: list[tuple[int, int]] = []
    keys = ["code_module", "code_presentation", "progress_bucket"]
    for _, group in frame.groupby(keys, sort=True):
        positive = group.index[group.target.eq(1)].to_numpy()
        negative = group.index[group.target.eq(0)].to_numpy()
        if not len(positive) or not len(negative):
            continue
        negative_probability = frame.loc[negative, "teacher_probability"].to_numpy()
        for positive_index in positive:
            distance = np.abs(
                negative_probability - float(frame.loc[positive_index, "teacher_probability"])
            )
            nearest = int(negative[int(np.argmin(distance))])
            pairs.append((int(positive_index), nearest))
    return np.asarray(pairs, dtype=np.int64)


def _pair_loader(
    inputs: OULADInputsV51,
    pairs: np.ndarray,
    *,
    batch_size: int,
    seed: int,
) -> DataLoader:
    positive, negative = pairs[:, 0], pairs[:, 1]
    tensors: list[torch.Tensor] = []
    for indices in (positive, negative):
        tensors.extend(
            [
                torch.from_numpy(inputs.sequence[indices]),
                torch.from_numpy(inputs.lengths[indices]),
                torch.from_numpy(inputs.mask[indices]),
                torch.from_numpy(inputs.aggregate[indices]),
                torch.from_numpy(inputs.static[indices]),
            ]
        )
    return DataLoader(
        TensorDataset(*tensors),
        batch_size=batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(seed),
        num_workers=0,
    )


def _predict(
    model: V6RiskRanking,
    inputs: OULADInputsV51,
    targets: TemporalTargets,
    indices: np.ndarray,
    device: torch.device,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    binary: list[np.ndarray] = []
    hazard: list[np.ndarray] = []
    outcome: list[np.ndarray] = []
    ranking: list[np.ndarray] = []
    with torch.no_grad():
        for batch in _loader(
            inputs, targets, indices, batch_size=batch_size, shuffle=False, seed=0
        ):
            values = [value.to(device) for value in batch[:5]]
            output = model(*values)
            binary.append(torch.sigmoid(output["binary_logit"]).cpu().numpy())
            hazard.append(torch.sigmoid(output["hazard_logit"]).cpu().numpy())
            outcome.append(torch.softmax(output["outcome_logit"], dim=1).cpu().numpy())
            ranking.append(output["ranking_score"].cpu().numpy())
    return tuple(
        np.concatenate(values) for values in (binary, hazard, outcome, ranking)
    )  # type: ignore[return-value]


def fit_ranking(
    train: OULADInputsV51,
    evaluation: OULADInputsV51,
    targets: TemporalTargets,
    train_indices: np.ndarray,
    evaluation_indices: np.ndarray,
    pairs: np.ndarray,
    *,
    config: dict[str, Any],
    initial_state: dict[str, torch.Tensor],
    ranking_weight: float,
    multitask_weights: dict[str, float],
    seed: int = 42,
    epochs: int = 8,
    device_name: str = "cuda",
) -> RankingFit:
    started = time.perf_counter()
    deterministic_seed(seed)
    device = torch.device(
        device_name if device_name != "cuda" or torch.cuda.is_available() else "cpu"
    )
    model = V6RiskRanking(
        train.sequence.shape[2], train.aggregate.shape[1], train.static.shape[1], config
    ).to(device)
    missing, unexpected = model.load_state_dict(initial_state, strict=False)
    if set(missing) != {"ranking_head.weight", "ranking_head.bias"} or unexpected:
        raise RuntimeError(f"Invalid ranking warm start: missing={missing}, unexpected={unexpected}")
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config.get("learning_rate", 1e-3)),
        weight_decay=float(config.get("weight_decay", 1e-5)),
    )
    batch_size = int(config.get("batch_size", 256))
    best_loss = float("inf")
    best_state = copy.deepcopy(model.state_dict())
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
            valid = outcome_target >= 0
            outcome_loss = nn.functional.cross_entropy(
                output["outcome_logit"][valid], outcome_target[valid].long()
            )
            loss = (
                binary_loss
                + float(multitask_weights["survival"]) * survival_loss
                + float(multitask_weights["outcome"]) * outcome_loss
            )
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), float(config["gradient_clip"]))
            optimizer.step()
            losses.append(float(loss.detach()))
        for pair_batch in _pair_loader(
            train, pairs, batch_size=batch_size, seed=seed + 100 + epoch
        ):
            pair_values = [value.to(device) for value in pair_batch]
            optimizer.zero_grad(set_to_none=True)
            positive_score = model(*pair_values[:5])["ranking_score"]
            negative_score = model(*pair_values[5:])["ranking_score"]
            pair_loss = nn.functional.softplus(-(positive_score - negative_score)).mean()
            weighted_pair_loss = float(ranking_weight) * pair_loss
            weighted_pair_loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), float(config["gradient_clip"]))
            optimizer.step()
            losses.append(float(weighted_pair_loss.detach()))
        mean_loss = float(np.mean(losses))
        if mean_loss < best_loss:
            best_loss = mean_loss
            best_state = copy.deepcopy(model.state_dict())
    model.load_state_dict(best_state)
    binary, hazard, outcome, ranking = _predict(
        model, evaluation, targets, evaluation_indices, device, batch_size
    )
    cpu_state = {name: value.detach().cpu().clone() for name, value in best_state.items()}
    replay = V6RiskRanking(
        train.sequence.shape[2], train.aggregate.shape[1], train.static.shape[1], config
    ).to(device)
    replay.load_state_dict(cpu_state)
    replay_binary, _, _, replay_ranking = _predict(
        replay, evaluation, targets, evaluation_indices, device, batch_size
    )
    replay_difference = max(
        float(np.max(np.abs(binary - replay_binary))),
        float(np.max(np.abs(ranking - replay_ranking))),
    )
    return RankingFit(
        binary_probability=binary,
        hazard_probability=hazard,
        outcome_probability=outcome,
        ranking_score=ranking,
        state_dict=cpu_state,
        checkpoint_sha256=_state_hash(cpu_state),
        replay_max_abs_difference=replay_difference,
        parameter_count=count_parameters(model),
        runtime_seconds=time.perf_counter() - started,
        pair_count=len(pairs),
    )


def ranking_metrics(target: np.ndarray, score: np.ndarray) -> dict[str, float]:
    order = np.argsort(-score, kind="stable")
    total_positive = max(1, int(target.sum()))
    result: dict[str, float] = {}
    for fraction in (0.05, 0.10, 0.20):
        count = max(1, int(np.ceil(len(target) * fraction)))
        selected = target[order[:count]]
        label = int(fraction * 100)
        result[f"precision_at_{label}_percent"] = float(selected.mean())
        result[f"recall_at_{label}_percent"] = float(selected.sum() / total_positive)
    for fraction in (0.10, 0.20):
        count = max(1, int(np.ceil(len(target) * fraction)))
        relevance = target[order[:count]].astype(float)
        discount = 1.0 / np.log2(np.arange(2, count + 2))
        dcg = float(np.sum(relevance * discount))
        ideal = np.sort(target)[::-1][:count].astype(float)
        idcg = float(np.sum(ideal * discount))
        result[f"ndcg_at_{int(fraction * 100)}_percent"] = dcg / idcg if idcg else 0.0
    return result


def screen_ranking(device_name: str = "cuda") -> dict[str, Any]:
    output = RANKING_ROOT / "gate.json"
    if output.is_file():
        result = json.loads(output.read_text(encoding="utf-8"))
        if result.get("status") == "COMPLETE":
            return result
    protocol = load_protocol()
    multitask_gate = json.loads(
        (ARTIFACT_ROOT / "prediction/multitask/gate.json").read_text(encoding="utf-8")
    )
    if not multitask_gate.get("gate_pass"):
        result = {
            "schema_version": "v6_ranking_gate_v1",
            "status": "COMPLETE",
            "selected": "B_MINIMAL_PRETRAINING",
            "gate_pass": False,
            "skip_reason": "SKIPPED_MULTITASK_GATE_FAILED",
            "outer_test_accessed": False,
            "future_accessed": False,
        }
        atomic_json(output, result)
        return result
    selected_multitask = str(multitask_gate["selected"])
    multitask_weights = multitask_gate["candidates"][selected_multitask]["weights"]
    _, v4_protocol, data = _load()
    targets = build_temporal_targets(data)
    splits = _inner_splits(data, 0, v4_protocol)
    config = _config()
    teacher = pd.read_parquet(
        ARTIFACT_ROOT.parent / "v5_4/oulad/teacher_oof_predictions.parquet"
    ).set_index("record_id")
    weights = list(protocol["ranking"]["registered_weights"])
    metrics_path = RANKING_ROOT / "fold_metrics.json"
    predictions_path = RANKING_ROOT / "inner_oof.parquet"
    rows: list[dict[str, Any]] = (
        pd.read_json(metrics_path).to_dict(orient="records")
        if metrics_path.is_file()
        else []
    )
    prediction_rows: list[dict[str, Any]] = (
        pd.read_parquet(predictions_path).to_dict(orient="records")
        if predictions_path.is_file()
        else []
    )
    completed = {(str(row["candidate"]), int(row["inner_fold"])) for row in rows}
    checkpoint_root = RANKING_ROOT / "checkpoints"
    checkpoint_root.mkdir(parents=True, exist_ok=True)
    for inner_fold, (train_index, validation_index) in enumerate(splits):
        train = prepare_oulad_inputs(data, train_index, train_index)
        validation = prepare_oulad_inputs(
            data, train_index, validation_index, fitted=train.preprocessors
        )
        train_record_ids = data.base.record_ids[train_index].astype(str)
        teacher_probability = teacher.loc[train_record_ids, "teacher_probability"].to_numpy()
        pairs = _pair_indices(
            data.base.cohort.iloc[train_index], data.y[train_index], teacher_probability
        )
        initial_state = torch.load(
            ARTIFACT_ROOT
            / f"prediction/multitask/checkpoints/{selected_multitask}_inner_{inner_fold}_seed_42.pt",
            map_location="cpu",
            weights_only=True,
        )
        for ranking_index, ranking_weight in enumerate(weights):
            candidate = f"R{ranking_index}"
            if (candidate, inner_fold) in completed:
                continue
            fit = fit_ranking(
                train,
                validation,
                targets,
                train_index,
                validation_index,
                pairs,
                config=config,
                initial_state=initial_state,
                ranking_weight=float(ranking_weight),
                multitask_weights=multitask_weights,
                device_name=device_name,
            )
            torch.save(
                fit.state_dict,
                checkpoint_root / f"{candidate}_inner_{inner_fold}_seed_42.pt",
            )
            frame = pd.DataFrame(
                {
                    "record_id": data.base.record_ids[validation_index].astype(str),
                    "inner_fold": inner_fold,
                    "target": data.y[validation_index].astype(int),
                    "probability": fit.binary_probability,
                    "ranking_score": fit.ranking_score,
                    "withdrawal_event": targets.withdrawal_event[validation_index],
                    "observation_week": targets.observation_week[validation_index],
                    "outcome_target": targets.outcome_target[validation_index],
                    "withdrawal_day": targets.withdrawal_day[validation_index],
                    "cutoff_day": data.base.cohort.iloc[validation_index].cutoff_day.to_numpy(),
                }
            )
            metrics = {
                **_candidate_metrics(frame, fit.hazard_probability, fit.outcome_probability),
                **ranking_metrics(frame.target.to_numpy(), fit.ranking_score),
            }
            rows.append(
                {
                    "candidate": candidate,
                    "ranking_weight": ranking_weight,
                    "inner_fold": inner_fold,
                    **metrics,
                    "pair_count": fit.pair_count,
                    "parameter_count": fit.parameter_count,
                    "runtime_seconds": fit.runtime_seconds,
                    "checkpoint_sha256": fit.checkpoint_sha256,
                    "replay_max_abs_difference": fit.replay_max_abs_difference,
                }
            )
            prediction_rows.extend(
                {"candidate": candidate, **row}
                for row in frame.to_dict(orient="records")
            )
        pd.DataFrame(rows).to_json(metrics_path, orient="records", indent=2)
        pd.DataFrame(prediction_rows).to_parquet(predictions_path, index=False)
    metrics = pd.DataFrame(rows)
    aggregate = metrics.groupby("candidate").mean(numeric_only=True)
    baseline_predictions = pd.read_parquet(
        ARTIFACT_ROOT / "prediction/multitask/inner_oof.parquet"
    )
    baseline_predictions = baseline_predictions[
        baseline_predictions.candidate.eq(selected_multitask)
    ]
    baseline_ranking = ranking_metrics(
        baseline_predictions.target.to_numpy(dtype=int),
        baseline_predictions.probability.to_numpy(dtype=float),
    )
    baseline = {
        **multitask_gate["candidates"][selected_multitask],
        **baseline_ranking,
    }
    candidates: dict[str, Any] = {}
    for candidate, row in aggregate.iterrows():
        raw_values = row.to_dict()
        values = {
            key: None if pd.isna(value) else value for key, value in raw_values.items()
        }
        guardrail = bool(
            raw_values["macro_f1"] >= baseline["macro_f1"] - 0.001
            and raw_values["at_risk_recall"] >= baseline["at_risk_recall"] - 0.002
            and raw_values["brier"] <= baseline["brier"] + 0.002
        )
        ranking_gain = bool(
            raw_values["pr_auc"] > baseline["pr_auc"]
            or raw_values["recall_at_10_percent"] > baseline["recall_at_10_percent"]
        )
        candidates[candidate] = {
            **values,
            "ranking_weight": weights[int(candidate[1:])],
            "guardrail_pass": guardrail,
            "ranking_gain": ranking_gain,
            "gate_pass": guardrail and ranking_gain,
        }
    passing = [name for name, row in candidates.items() if row["gate_pass"]]
    selected = (
        max(
            passing,
            key=lambda name: (
                candidates[name]["macro_f1"],
                candidates[name]["recall_at_10_percent"],
            ),
        )
        if passing
        else None
    )
    result = {
        "schema_version": "v6_ranking_gate_v1",
        "status": "COMPLETE",
        "selected": selected if selected else "C_TEMPORAL_MULTITASK",
        "selected_multitask": selected_multitask,
        "gate_pass": selected is not None,
        "candidates": candidates,
        "baseline": baseline,
        "pair_contract": "same_module_presentation_progress_near_crossfit_teacher_probability",
        "outer_test_accessed": False,
        "future_accessed": False,
    }
    atomic_json(output, result)
    table = "\n".join(
        f"| {name} | {row['ranking_weight']:.2f} | {row['macro_f1']:.6f} | "
        f"{row['pr_auc']:.6f} | {row['recall_at_10_percent']:.6f} | "
        f"{str(row['gate_pass']).lower()} |"
        for name, row in candidates.items()
    )
    atomic_text(
        REPORT_ROOT / "RISK_RANKING_REPORT.md",
        f"""# V6 risk-ranking report

| Candidate | Lambda | Macro-F1 | PR-AUC | Recall@10% | Gate |
|---|---:|---:|---:|---:|---:|
{table}

Selected: **{result['selected']}**. Pairs use only records in the same module,
presentation and course-progress bucket. Near-probability matching uses the
cross-fitted V5.4 XGBoost teacher on outer-training fold 0. No outer-test or
Future OULAD record was accessed.
""",
    )
    return result


__all__ = ["V6RiskRanking", "fit_ranking", "ranking_metrics", "screen_ranking"]
