"""Cutoff-safe AP fine-tuning for the frozen Phase 7 A2 topology."""
from __future__ import annotations

import copy
import time

import numpy as np
import torch
from libauc.losses import APLoss
from libauc.optimizers import SOAP
from sklearn.metrics import average_precision_score

from src.hybrid.optimization.phase6b import stage_threshold_metrics
from src.hybrid.optimization.phase6f import positive_safe_batches
from src.hybrid.phase7.execution import (
    STAGES, _model, _partitions, _scale, oulad_outcomes, phase7_domain,
)
from src.hybrid.training.data import sample_prefixes
from src.hybrid.training.trainer import seed_everything


def fine_tune_a2_from_checkpoint(domain, fold, seed, bce_config, ap_config, checkpoint_path):
    """Restore a matched BCE checkpoint, optimize AP on FIT, select on STOP, evaluate VALID."""
    views, context, numeric, categorical = phase7_domain(domain)
    fit, stop, valid = _partitions(domain, fold, context)
    static_map, prep = _scale(views, context, numeric, categorical, fit, domain)
    indices = {s: {r: i for i, r in enumerate(v.record_id.astype(str))} for s, v in views.items()}
    temporal_dim = next(iter(views.values())).temporal.shape[2]
    aggregate_dim = next(iter(views.values())).aggregate.shape[1]
    seed_everything(seed)
    model = _model(domain, "A2", prep.output_dim, temporal_dim, aggregate_dim, bce_config["dropout"])
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if payload.get("domain") != domain or payload.get("candidate") != "A2" or int(payload.get("fold")) != int(fold) or int(payload.get("seed")) != int(seed) or payload.get("training_config") != bce_config:
        raise RuntimeError("PHASE7E_BCE_CHECKPOINT_IDENTITY_MISMATCH")
    model.load_state_dict(payload["model_state"])
    parameter_count = sum(p.numel() for p in model.parameters())
    torch.cuda.reset_peak_memory_stats()
    started = time.monotonic()

    ap_lr = float(bce_config["learning_rate"]) * float(ap_config["lr_multiplier"])
    optimizer = SOAP(model.parameters(), lr=ap_lr, mode="adam",
                     clip_value=float(bce_config["gradient_clip_norm"]),
                     weight_decay=float(bce_config["weight_decay"]), device="cuda", verbose=False)
    fit_by_stage = {s: [r for r in fit if r in indices[s]] for s in STAGES[domain]}
    stage_position = {s: {r: i for i, r in enumerate(ids)} for s, ids in fit_by_stage.items()}
    ap_losses = {s: APLoss(data_len=len(ids), gamma=.9, margin=1.0, device="cuda") for s, ids in fit_by_stage.items()}
    unique_target = np.asarray([next(v.target[indices[s][r]] for s, v in views.items() if r in indices[s]) for r in fit])
    base_weight = (len(unique_target) - unique_target.sum()) / max(1, unique_target.sum())
    positive_weight = base_weight * float(bce_config.get("positive_class_weight_multiplier", 1.0))
    bce = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor([positive_weight], device="cuda"))

    def forward(stage, ids):
        view = views[stage]
        ix = np.asarray([indices[stage][r] for r in ids])
        static = torch.tensor(np.asarray([static_map[r] for r in ids]), dtype=torch.float32, device="cuda")
        temporal = torch.tensor(view.temporal[ix], dtype=torch.float32, device="cuda")
        mask = torch.tensor(view.temporal_mask[ix], device="cuda")
        lengths = torch.tensor(view.lengths[ix], device="cuda")
        aggregate = torch.tensor(view.aggregate[ix], dtype=torch.float32, device="cuda")
        available = torch.tensor(view.aggregate_available[ix], dtype=torch.float32, device="cuda")
        progress = torch.tensor(view.progress[ix], dtype=torch.float32, device="cuda")
        return model(static, temporal, mask, lengths, aggregate, available, progress), view.target[ix]

    @torch.no_grad()
    def predict(stage, ids):
        model.eval(); output = []
        batch_size = int(bce_config["batch_size"])
        for start in range(0, len(ids), batch_size):
            output.append(torch.sigmoid(forward(stage, ids[start:start + batch_size])[0]).float().cpu().numpy())
        return np.concatenate(output)

    best_score, best_epoch, best_state = -np.inf, 0, None
    for epoch in range(int(ap_config["epochs"])):
        model.train()
        if domain == "uci":
            target = np.asarray([views[STAGES[domain][0]].target[indices[STAGES[domain][0]][r]] for r in fit], dtype=int)
            for batch in positive_safe_batches(target, int(bce_config["batch_size"]), seed, epoch):
                ids = [fit[i] for i in batch]
                optimizer.zero_grad(set_to_none=True); losses = []
                for stage in STAGES[domain]:
                    logits, labels = forward(stage, ids)
                    sample_index = torch.as_tensor([stage_position[stage][r] for r in ids], dtype=torch.long, device="cuda")
                    target_tensor = torch.as_tensor(labels, dtype=torch.float32, device="cuda")
                    ap = ap_losses[stage](torch.sigmoid(logits.float()), target_tensor, sample_index)
                    losses.append(ap if ap_config["mix"] == "ap_only" else .5 * ap + .5 * bce(logits.float(), target_tensor))
                loss = torch.stack(losses).mean()
                if not torch.isfinite(loss): raise RuntimeError("NONFINITE_AP_LOSS")
                loss.backward(); norm = torch.nn.utils.clip_grad_norm_(model.parameters(), float(bce_config["gradient_clip_norm"]))
                if not torch.isfinite(norm): raise RuntimeError("NONFINITE_AP_GRADIENT")
                optimizer.step()
        else:
            available = {r: [s for s in STAGES[domain] if r in indices[s]] for r in fit}
            chosen = sample_prefixes(fit, [available[r] for r in fit], seed, epoch)
            by_stage = {s: [] for s in STAGES[domain]}
            for record_id, stage in zip(fit, chosen): by_stage[stage].append(record_id)
            for stage, ids in by_stage.items():
                labels = np.asarray([views[stage].target[indices[stage][r]] for r in ids], dtype=int)
                for batch in positive_safe_batches(labels, int(bce_config["batch_size"]), seed + STAGES[domain].index(stage) * 100003, epoch):
                    batch_ids = [ids[i] for i in batch]
                    optimizer.zero_grad(set_to_none=True)
                    logits, batch_target = forward(stage, batch_ids)
                    sample_index = torch.as_tensor([stage_position[stage][r] for r in batch_ids], dtype=torch.long, device="cuda")
                    target_tensor = torch.as_tensor(batch_target, dtype=torch.float32, device="cuda")
                    ap = ap_losses[stage](torch.sigmoid(logits.float()), target_tensor, sample_index)
                    loss = ap if ap_config["mix"] == "ap_only" else .5 * ap + .5 * bce(logits.float(), target_tensor)
                    if not torch.isfinite(loss): raise RuntimeError("NONFINITE_AP_LOSS")
                    loss.backward(); norm = torch.nn.utils.clip_grad_norm_(model.parameters(), float(bce_config["gradient_clip_norm"]))
                    if not torch.isfinite(norm): raise RuntimeError("NONFINITE_AP_GRADIENT")
                    optimizer.step()
        stop_scores = []
        for stage, view in views.items():
            ids = [r for r in stop if r in indices[stage]]
            target = view.target[[indices[stage][r] for r in ids]]
            stop_scores.append(average_precision_score(target, predict(stage, ids)))
        score = float(np.mean(stop_scores))
        if score > best_score:
            best_score, best_epoch = score, epoch + 1
            best_state = copy.deepcopy({k: value.detach().cpu() for k, value in model.state_dict().items()})
    if best_state is None: raise RuntimeError("NO_FINITE_AP_EPOCH")
    model.load_state_dict(best_state)

    metrics = {}
    for stage, view in views.items():
        train_ids = fit_by_stage[stage]
        stop_ids = [r for r in stop if r in indices[stage]]
        valid_ids = [r for r in valid if r in indices[stage]]
        train_score = predict(stage, train_ids); stop_score = predict(stage, stop_ids); valid_score = predict(stage, valid_ids)
        train_target = view.target[[indices[stage][r] for r in train_ids]]
        stop_target = view.target[[indices[stage][r] for r in stop_ids]]
        valid_index = [indices[stage][r] for r in valid_ids]; valid_target = view.target[valid_index]
        selected = stage_threshold_metrics(stop_target, stop_score, valid_target, valid_score)
        row = selected["stop_selected"]
        row.update({"train_pr_auc": float(average_precision_score(train_target, train_score)),
                    "validation_pr_auc": row["pr_auc"],
                    "train_validation_gap": float(average_precision_score(train_target, train_score) - row["pr_auc"]),
                    "selected_threshold": selected["selected_threshold"], "threshold_source": "stop_only"})
        if domain == "oulad":
            outcomes = oulad_outcomes(); prediction = valid_score >= selected["selected_threshold"]
            labels = np.asarray([outcomes[r] for r in valid_ids])
            row["recall_fail"] = float(prediction[labels == "Fail"].mean()) if np.any(labels == "Fail") else None
            row["recall_withdrawn"] = float(prediction[labels == "Withdrawn"].mean()) if np.any(labels == "Withdrawn") else None
            row["risk_prevalence"] = float(valid_target.mean())
        metrics[stage] = row
    return {"metrics": metrics, "parameter_count": parameter_count, "best_epoch": best_epoch,
            "bce_best_epoch": int(payload["best_epoch"]), "runtime_seconds": time.monotonic() - started,
            "peak_gpu_memory_bytes": int(torch.cuda.max_memory_allocated()), "outer_test_used": False}
