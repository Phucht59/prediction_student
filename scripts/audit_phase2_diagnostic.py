"""Controlled inner-only OULAD learning-trajectory diagnostic for Phase 2."""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    log_loss,
    precision_recall_fscore_support,
    roc_auc_score,
)
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.pipelines import oulad  # noqa: E402
from src.training.control import select_refit_epoch, select_research_threshold  # noqa: E402


OUT = ROOT / "artifacts" / "audit" / "phase2"
PREREGISTERED = {
    "status": "DIAGNOSTIC_ONLY",
    "dataset": "oulad",
    "model_family": "cnn_bilstm",
    "outer_fold": 0,
    "inner_folds": [0, 1],
    "seed": 42,
    "epochs": list(range(1, 31)),
    "monitor_policy": "mean_stage_validation_nll",
    "monitor_mode": "min",
    "refit_epoch_policy": "median",
    "outer_labels_used": False,
    "outer_metrics_reported": False,
}


def _ece(y: np.ndarray, p: np.ndarray, bins: int = 15) -> float:
    edges = np.linspace(0, 1, bins + 1)
    result = 0.0
    for low, high in zip(edges[:-1], edges[1:], strict=True):
        included = (p >= low) & (p < (high if high < 1 else high + 1e-9))
        if included.any():
            result += included.mean() * abs(p[included].mean() - y[included].mean())
    return float(result)


def _stage_metrics(y: np.ndarray, p: np.ndarray) -> dict[str, float]:
    clipped = np.clip(p, 1e-7, 1 - 1e-7)
    research = select_research_threshold(y, clipped)
    fixed = clipped >= 0.5
    selected = clipped >= research["threshold"]
    precision, recall = precision_recall_fscore_support(
        y, fixed, average="binary", zero_division=0
    )[:2]
    selected_precision, selected_recall = precision_recall_fscore_support(
        y, selected, average="binary", zero_division=0
    )[:2]
    return {
        "validation_bce": float(log_loss(y, clipped, labels=[0, 1])),
        "validation_nll": float(log_loss(y, clipped, labels=[0, 1])),
        "macro_f1_at_0_5": float(f1_score(y, fixed, average="macro")),
        "research_threshold": float(research["threshold"]),
        "threshold_optimized_macro_f1": float(research["macro_f1"]),
        "pr_auc": float(average_precision_score(y, clipped)),
        "roc_auc": float(roc_auc_score(y, clipped)),
        "brier": float(np.mean((clipped - y) ** 2)),
        "ece": _ece(y, clipped),
        "risk_precision_at_0_5": float(precision),
        "risk_recall_at_0_5": float(recall),
        "risk_precision_at_research_threshold": float(selected_precision),
        "risk_recall_at_research_threshold": float(selected_recall),
        "probability_mean": float(clipped.mean()),
        "probability_std": float(clipped.std()),
        "positive_rate": float(y.mean()),
        "eligible_count": int(len(y)),
    }


def _train_trajectory(
    train: tuple, validation: tuple, *, inner_fold: int
) -> list[dict[str, float | int | str | bool]]:
    seed = int(PREREGISTERED["seed"])
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    frame, seq, length, mask, aggregate, y, weight = train
    val_frame, val_seq, val_length, val_mask, val_aggregate, val_y, _ = validation
    preprocessor = oulad._DeepPreprocessor().fit(frame, aggregate)
    aggregate, static = preprocessor.transform(frame, aggregate)
    val_aggregate, val_static = preprocessor.transform(val_frame, val_aggregate)
    config = oulad._deep_config(oulad._protocol())
    device = torch.device("cuda")
    model = oulad._deep_model("cnn_bilstm", aggregate.shape[1], static.shape[1], config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["learning_rate"]),
        weight_decay=float(config["weight_decay"]),
    )
    positive_weight = float((y == 0).sum() / max((y == 1).sum(), 1))
    risk_loss = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor(positive_weight, device=device), reduction="none"
    )
    dataset = TensorDataset(
        torch.from_numpy(seq),
        torch.from_numpy(length.astype(np.int64)),
        torch.from_numpy(mask.astype(np.float32)),
        torch.from_numpy(aggregate),
        torch.from_numpy(static),
        torch.from_numpy(y.astype(np.float32)),
        torch.from_numpy(weight.astype(np.float32)),
        torch.from_numpy(frame.outcome_aux.to_numpy(dtype=np.int64)),
        torch.from_numpy(frame.cutoff_day.to_numpy(dtype=np.int64)),
        torch.from_numpy(frame.module_presentation_length.to_numpy(dtype=np.int64)),
        torch.from_numpy(
            frame.date_unregistration.fillna(-1).to_numpy(dtype=np.int64)
        ),
    )
    loader = DataLoader(
        dataset,
        batch_size=int(config["batch_size"]),
        shuffle=True,
        generator=torch.Generator().manual_seed(seed),
    )
    rows: list[dict[str, float | int | str | bool]] = []
    for epoch in PREREGISTERED["epochs"]:
        started = time.perf_counter()
        model.train()
        losses: list[float] = []
        gradient_norms: list[float] = []
        for batch in loader:
            (
                batch_seq,
                batch_length,
                batch_mask,
                batch_aggregate,
                batch_static,
                target,
                sample_weight,
                outcome,
                cutoff,
                course_end,
                unregistration,
            ) = (value.to(device) for value in batch)
            optimizer.zero_grad()
            output = model(
                batch_seq,
                batch_length,
                batch_mask,
                batch_aggregate,
                batch_static,
            )
            loss, _ = oulad._multitask_loss(
                output,
                target,
                sample_weight,
                outcome,
                cutoff,
                course_end,
                unregistration,
                risk_loss,
                survival_weight=float(config["survival_weight"]),
                outcome_weight=float(config["outcome_weight"]),
            )
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), float(config["gradient_clip_norm"])
            )
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
            gradient_norms.append(float(norm.detach().cpu()))
        probability = oulad._predict_deep(
            model,
            val_seq,
            val_length,
            val_mask,
            val_aggregate,
            val_static,
            "cnn_bilstm",
            device,
        )
        for stage in oulad.STAGES:
            selected = val_frame.prediction_stage.eq(stage).to_numpy()
            metrics = _stage_metrics(val_y[selected], probability[selected])
            rows.append(
                {
                    "diagnostic_status": "DIAGNOSTIC_ONLY",
                    "outer_fold": 0,
                    "inner_fold": inner_fold,
                    "seed": seed,
                    "epoch": int(epoch),
                    "prediction_stage": stage,
                    "train_loss": float(np.mean(losses)),
                    "gradient_norm": float(np.mean(gradient_norms)),
                    "learning_rate": float(optimizer.param_groups[0]["lr"]),
                    "epoch_runtime_seconds": time.perf_counter() - started,
                    "outer_labels_used": False,
                    **metrics,
                }
            )
        if epoch in {1, 4, 10, 15, 20, 30}:
            print(f"inner_fold={inner_fold} epoch={epoch} complete", flush=True)
    return rows


def _best_epoch(frame: pd.DataFrame, metric: str, mode: str) -> int:
    by_epoch = frame.groupby("epoch", as_index=False)[metric].mean()
    ascending = mode == "min"
    return int(
        by_epoch.sort_values(
            [metric, "epoch"], ascending=[ascending, True], kind="stable"
        ).iloc[0]["epoch"]
    )


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("BLOCKED_GPU: Phase 2 diagnostic requires CUDA")
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "diagnostic_preregistration.json").write_text(
        json.dumps(PREREGISTERED, indent=2) + "\n", encoding="utf-8"
    )
    bundle = oulad._build_bundle()
    base = bundle.base[
        ["base_record_id", "id_student", "outer_fold", "target"]
    ].drop_duplicates()
    rows: list[dict[str, float | int | str | bool]] = []
    for inner_fold, (fit_ids, validation_ids) in enumerate(
        oulad._inner_splits(base, 0)
    ):
        rows.extend(
            _train_trajectory(
                oulad._stage_rows(bundle, fit_ids),
                oulad._stage_rows(bundle, validation_ids),
                inner_fold=inner_fold,
            )
        )
    stage_curve = pd.DataFrame(rows)
    stage_curve.to_csv(OUT / "stage_learning_curve.csv", index=False)
    numeric = [
        column
        for column in stage_curve.select_dtypes(include=[np.number]).columns
        if column not in {"outer_fold", "inner_fold", "seed", "epoch"}
    ]
    epoch_curve = (
        stage_curve.groupby("epoch", as_index=False)[numeric]
        .mean()
        .assign(
            diagnostic_status="DIAGNOSTIC_ONLY",
            outer_fold=0,
            seed=42,
            inner_fold_count=2,
            outer_labels_used=False,
        )
    )
    epoch_curve.to_csv(OUT / "epoch_learning_curve.csv", index=False)

    policies = [
        ("fixed_macro_f1_at_0_5", "macro_f1_at_0_5", "max"),
        ("validation_nll", "validation_nll", "min"),
        ("validation_pr_auc", "pr_auc", "max"),
        (
            "threshold_optimized_macro_f1",
            "threshold_optimized_macro_f1",
            "max",
        ),
    ]
    comparisons: list[dict[str, object]] = []
    selected_by_policy: dict[str, list[int]] = {}
    for policy, metric, mode in policies:
        epochs: list[int] = []
        for inner_fold, fold_rows in stage_curve.groupby("inner_fold"):
            selected_epoch = _best_epoch(fold_rows, metric, mode)
            epochs.append(selected_epoch)
            comparisons.append(
                {
                    "diagnostic_status": "DIAGNOSTIC_ONLY",
                    "policy": policy,
                    "metric": metric,
                    "mode": mode,
                    "inner_fold": int(inner_fold),
                    "selected_epoch": selected_epoch,
                    "outer_labels_used": False,
                }
            )
        selected_by_policy[policy] = epochs
        comparisons.append(
            {
                "diagnostic_status": "DIAGNOSTIC_ONLY",
                "policy": policy,
                "metric": metric,
                "mode": mode,
                "inner_fold": "AGGREGATED_MEDIAN",
                "selected_epoch": select_refit_epoch(epochs),
                "outer_labels_used": False,
            }
        )
    pd.DataFrame(comparisons).to_csv(
        OUT / "checkpoint_policy_comparison.csv", index=False
    )
    selected_epoch = select_refit_epoch(selected_by_policy["validation_nll"])
    selection = {
        **PREREGISTERED,
        "per_policy_inner_selected_epochs": selected_by_policy,
        "recommended_checkpoint_policy": "mean_stage_validation_nll",
        "selected_inner_epochs": selected_by_policy["validation_nll"],
        "propagated_fixed_refit_epoch": selected_epoch,
        "aggregation": "round_half_up_median",
        "selection_scope": "inner_validation_only",
    }
    (OUT / "epoch_selection.json").write_text(
        json.dumps(selection, indent=2) + "\n", encoding="utf-8"
    )
    print(f"propagated_fixed_refit_epoch={selected_epoch}", flush=True)


if __name__ == "__main__":
    main()
