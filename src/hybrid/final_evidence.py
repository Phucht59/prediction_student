"""Evaluation-only helpers for the frozen Phase 7 A2/BCE model."""
from __future__ import annotations

import copy
import time
from types import MethodType

import numpy as np
import torch
from sklearn.metrics import average_precision_score

from src.hybrid.optimization.phase6b import stage_threshold_metrics
from src.hybrid.phase7.execution import STAGES, _model, _partitions, _scale, oulad_outcomes, phase7_domain


ABLATION_BRANCHES = {
    "B0_static": ("static",),
    "B1_cnn": ("cnn",),
    "B2_bilstm": ("bilstm",),
    "B3_cnn_bilstm": ("cnn", "bilstm"),
    "B4_static_temporal": ("static", "cnn", "bilstm"),
    "B5_final": ("static", "cnn", "bilstm", "aggregate"),
}


def apply_branch_ablation(model, active_branches):
    """Mask semantic paths at A2 residual fusion without changing parameters or branch topology."""
    active = frozenset(active_branches)
    if not active or not active.issubset({"static", "cnn", "bilstm", "aggregate"}):
        raise ValueError(f"invalid branch ablation: {sorted(active)}")

    def forward(self, static, temporal, temporal_mask, lengths, aggregate, aggregate_available, progress):
        hs, hc, hl, ha = self.representations(static, temporal, temporal_mask, lengths, aggregate, aggregate_available)
        temporal_available = (lengths > 0).to(hs.dtype)
        availability = torch.stack((
            torch.ones_like(temporal_available) if "static" in active else torch.zeros_like(temporal_available),
            temporal_available if "cnn" in active else torch.zeros_like(temporal_available),
            temporal_available if "bilstm" in active else torch.zeros_like(temporal_available),
            aggregate_available.to(hs.dtype) if "aggregate" in active else torch.zeros_like(temporal_available),
        ), -1)
        stacked = torch.stack((hs, hc, hl, ha), 1)
        base = (stacked * availability.unsqueeze(-1)).sum(1) / availability.sum(1, keepdim=True).clamp_min(1.)
        correction = torch.zeros_like(base)
        self.last_diagnostics = {"h_static": hs.detach(), "h_cnn": hc.detach(),
                                 "h_bilstm": hl.detach(), "h_aggregate": ha.detach(),
                                 "base": base.detach(), "interaction": correction.detach()}
        return self.head(self.fusion_norm(base)).squeeze(-1)

    model.forward = MethodType(forward, model)
    model.active_ablation_branches = tuple(sorted(active))
    return model


def evaluate_a2_checkpoint(domain, fold, seed, config, checkpoint_path):
    """Evaluate a matched frozen BCE checkpoint and retain paired VALID predictions."""
    views, context, numeric, categorical = phase7_domain(domain)
    fit, stop, valid = _partitions(domain, fold, context)
    static_map, prep = _scale(views, context, numeric, categorical, fit, domain)
    indices = {s: {r: i for i, r in enumerate(v.record_id.astype(str))} for s, v in views.items()}
    temporal_dim = next(iter(views.values())).temporal.shape[2]
    aggregate_dim = next(iter(views.values())).aggregate.shape[1]
    model = _model(domain, "A2", prep.output_dim, temporal_dim, aggregate_dim, config["dropout"])
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if payload.get("domain") != domain or payload.get("candidate") != "A2" or int(payload.get("fold")) != int(fold) or int(payload.get("seed")) != int(seed) or payload.get("training_config") != config:
        raise RuntimeError("FINAL_BCE_CHECKPOINT_IDENTITY_MISMATCH")
    model.load_state_dict(payload["model_state"]); model.eval(); started = time.monotonic()

    def forward(stage, ids):
        view = views[stage]; ix = np.asarray([indices[stage][r] for r in ids])
        return model(
            torch.tensor(np.asarray([static_map[r] for r in ids]), dtype=torch.float32, device="cuda"),
            torch.tensor(view.temporal[ix], dtype=torch.float32, device="cuda"),
            torch.tensor(view.temporal_mask[ix], device="cuda"), torch.tensor(view.lengths[ix], device="cuda"),
            torch.tensor(view.aggregate[ix], dtype=torch.float32, device="cuda"),
            torch.tensor(view.aggregate_available[ix], dtype=torch.float32, device="cuda"),
            torch.tensor(view.progress[ix], dtype=torch.float32, device="cuda"))

    @torch.no_grad()
    def predict(stage, ids):
        output = []; batch_size = int(config["batch_size"])
        for start in range(0, len(ids), batch_size):
            output.append(torch.sigmoid(forward(stage, ids[start:start + batch_size])).float().cpu().numpy())
        return np.concatenate(output)

    metrics, predictions = {}, {}; outcomes = oulad_outcomes() if domain == "oulad" else {}
    for stage, view in views.items():
        train_ids = [r for r in fit if r in indices[stage]]; stop_ids = [r for r in stop if r in indices[stage]]
        valid_ids = [r for r in valid if r in indices[stage]]
        train_score = predict(stage, train_ids); stop_score = predict(stage, stop_ids); valid_score = predict(stage, valid_ids)
        train_target = view.target[[indices[stage][r] for r in train_ids]]
        stop_target = view.target[[indices[stage][r] for r in stop_ids]]
        valid_target = view.target[[indices[stage][r] for r in valid_ids]]
        selected = stage_threshold_metrics(stop_target, stop_score, valid_target, valid_score); row = selected["stop_selected"]
        row.update({"train_pr_auc": float(average_precision_score(train_target, train_score)),
                    "validation_pr_auc": row["pr_auc"],
                    "train_validation_gap": float(average_precision_score(train_target, train_score) - row["pr_auc"]),
                    "selected_threshold": selected["selected_threshold"], "threshold_source": "stop_only"})
        if domain == "oulad":
            labels = np.asarray([outcomes[r] for r in valid_ids]); positive = valid_score >= selected["selected_threshold"]
            row["recall_fail"] = float(positive[labels == "Fail"].mean())
            row["recall_withdrawn"] = float(positive[labels == "Withdrawn"].mean())
            row["risk_prevalence"] = float(valid_target.mean())
        metrics[stage] = row
        predictions[stage] = [{"record_id": record_id, "target": int(target), "score": float(score),
                               "prediction": int(score >= selected["selected_threshold"]),
                               "outcome": outcomes.get(record_id)}
                              for record_id, target, score in zip(valid_ids, valid_target, valid_score)]
    return {"metrics": metrics, "predictions": predictions, "best_epoch": int(payload["best_epoch"]),
            "parameter_count": sum(p.numel() for p in model.parameters()),
            "runtime_seconds": time.monotonic() - started, "peak_gpu_memory_bytes": int(torch.cuda.max_memory_allocated()),
            "outer_test_used": False}
