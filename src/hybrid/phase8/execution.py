"""Small, sealed-inner-development Phase8 screening runner.

This runner deliberately does not enumerate outer-test records.  It is a
screening instrument: candidates that survive here must still receive a
multi-fold/multi-seed recheck before any architecture-selection claim.
"""
from __future__ import annotations

import copy
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from src.hybrid.phase7.execution import STAGES, _pad, _partitions, _scale, oulad_outcomes, phase7_domain
from src.hybrid.data.oulad import OULAD_CATEGORICAL_CONTEXT, OULAD_NUMERIC_CONTEXT
from src.hybrid.phase8.final100 import FINAL_STAGE, load_unified_view
from src.hybrid.optimization.phase6b import stage_threshold_metrics
from src.hybrid.phase8.model import Phase8HybridConfig, Phase8UnifiedHybrid
from src.hybrid.phase8.residual import ResidualTemporalConfig, ResidualTemporalHybrid
from src.hybrid.phase8.data_variants import apply_data_variant, VARIANTS
from src.hybrid.training.data import sample_prefixes, sample_prefixes_stage_balanced
from src.hybrid.training.trainer import seed_everything

ROOT = Path(__file__).resolve().parents[3]


def _oulad_domain_with_final() -> tuple[dict, pd.DataFrame, list[str], list[str], dict]:
    """Combine cached FINAL-100 arrays with the existing Phase7 views.

    Materializing FINAL-100 separately avoids holding the 8.4M-row compact
    daily table alongside all endpoint tensors during training.
    """
    dev_dir = ROOT / "artifacts/hybrid/phase8/final_development/dev_views"
    manifest = dev_dir / "manifest.json"
    if not manifest.exists():
        raise RuntimeError("FINAL100_DEV_VIEWS_MISSING_RUN_MATERIALIZER")
    stages = json.loads(manifest.read_text(encoding="utf-8"))["stages"]
    views = {stage: load_unified_view(dev_dir / stage) for stage in stages}
    _pad(views)
    context = pd.read_parquet(dev_dir / "context.parquet")
    audit = json.loads((ROOT / "artifacts/hybrid/phase8/final_development/final100/data_audit.json").read_text(encoding="utf-8"))
    return views, context, OULAD_NUMERIC_CONTEXT, OULAD_CATEGORICAL_CONTEXT, audit


def run_phase8_screen(*, domain: str, protocol: str, temporal_variant: str, fold: int = 0, seed: int = 42,
                      max_epochs: int = 16, patience: int = 5, batch_size: int = 256,
                      fusion: str = "adaptive", entropy_floor_coefficient: float = 0.0,
                      branch_mode: str = "full", collect_forensics: bool = False,
                      model_family: str = "representation", residual_sample_gate: bool = True,
                      data_variant: str = "D0_raw", include_final100: bool = False,
                      final_only: bool = False) -> dict:
    """Run one controlled screen for a fixed data/fold/seed identity."""
    if protocol not in {"P0_random_prefix", "P1_stage_balanced", "P2_all_stage"}:
        raise ValueError("unknown training protocol")
    if temporal_variant not in {"raw", "exposure_normalized"}:
        raise ValueError("unknown temporal variant")
    if include_final100:
        if domain != "oulad":
            raise ValueError("FINAL-100 is OULAD-specific")
        views, context, numeric, categorical, final_audit = _oulad_domain_with_final()
        if final_only:
            views = {FINAL_STAGE: views[FINAL_STAGE]}
    else:
        views, context, numeric, categorical = phase7_domain(domain)
    if domain == "oulad":
        if data_variant not in VARIANTS: raise ValueError("unknown data_variant")
        views = {stage: apply_data_variant(view, data_variant) for stage, view in views.items()}
    elif data_variant != "D0_raw":
        raise ValueError("non-D0 data variants currently apply to OULAD only")
    if temporal_variant == "exposure_normalized":
        if domain != "oulad":
            raise ValueError("exposure-normalized ablation applies to OULAD only")
        for view in views.values():
            object.__setattr__(view, "temporal", view.temporal_exposure_normalized.copy())
    fit, stop, valid = _partitions(domain, fold, context)
    stages = tuple(views)
    static_map, prep = _scale(views, context, numeric, categorical, fit, domain)
    indexes = {stage: {record_id: i for i, record_id in enumerate(view.record_id.astype(str))} for stage, view in views.items()}
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seed_everything(seed)
    first = next(iter(views.values()))
    if model_family == "representation":
        model = Phase8UnifiedHybrid(Phase8HybridConfig(
            prep.output_dim, first.temporal.shape[2], first.aggregate.shape[1],
            fusion=fusion, entropy_floor_coefficient=entropy_floor_coefficient, branch_mode=branch_mode,
        )).to(device)
    elif model_family == "residual":
        if branch_mode != "full":
            raise ValueError("residual model currently requires the full temporal pair")
        model = ResidualTemporalHybrid(ResidualTemporalConfig(prep.output_dim, first.temporal.shape[2], first.aggregate.shape[1], sample_gate=residual_sample_gate)).to(device)
    else:
        raise ValueError("unknown model_family")
    optimizer = torch.optim.AdamW(model.parameters(), lr=8e-4, weight_decay=2e-4)
    y = context[context.record_id.astype(str).isin(fit)].target.to_numpy()
    pos_weight = torch.tensor([(len(y) - y.sum()) / max(1, y.sum())], device=device)
    loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    def batch_inputs(stage: str, record_ids: list[str]):
        view = views[stage]; ix = np.asarray([indexes[stage][record_id] for record_id in record_ids])
        return (
            torch.tensor(np.asarray([static_map[r] for r in record_ids]), dtype=torch.float32, device=device),
            torch.tensor(view.temporal[ix], dtype=torch.float32, device=device),
            torch.tensor(view.temporal_mask[ix], device=device), torch.tensor(view.lengths[ix], device=device),
            torch.tensor(view.aggregate[ix], dtype=torch.float32, device=device),
            torch.tensor(view.aggregate_available[ix], dtype=torch.float32, device=device),
            torch.tensor(view.progress[ix], dtype=torch.float32, device=device),
        )

    def forward(stage: str, record_ids: list[str], branch_mask=None):
        logits = model(*batch_inputs(stage, record_ids), branch_mask=branch_mask)
        return logits, views[stage].target[np.asarray([indexes[stage][record_id] for record_id in record_ids])]

    @torch.no_grad()
    def predict(stage: str, record_ids: list[str], branch_mask=None) -> np.ndarray:
        model.eval()
        outputs = []
        for begin in range(0, len(record_ids), batch_size):
            outputs.append(torch.sigmoid(forward(stage, record_ids[begin:begin + batch_size], branch_mask)[0]).cpu().numpy())
        return np.concatenate(outputs)

    best_score, best_epoch, stale, state = -np.inf, 0, 0, None
    started = time.monotonic()
    eligible = {record_id: [stage for stage in stages if record_id in indexes[stage]] for record_id in fit}
    for epoch in range(max_epochs):
        model.train()
        if protocol == "P2_all_stage":
            # A record contributes every stage it is actually eligible for;
            # later cutoffs have smaller historical cohorts after cutoff-safe
            # withdrawal filtering, so blindly using all FIT ids is invalid.
            batches = [(stage, [r for r in fit if r in indexes[stage]]) for stage in stages]
        else:
            choices = (sample_prefixes(fit, [eligible[r] for r in fit], seed, epoch) if protocol == "P0_random_prefix"
                       else sample_prefixes_stage_balanced(fit, [eligible[r] for r in fit], seed, epoch))
            batches = [(stage, [r for r, selected in zip(fit, choices) if selected == stage]) for stage in stages]
        for stage, ids in batches:
            for begin in range(0, len(ids), batch_size):
                chunk = ids[begin:begin + batch_size]
                if not chunk:
                    continue
                optimizer.zero_grad(set_to_none=True)
                logits, labels = forward(stage, chunk)
                loss = loss_fn(logits, torch.tensor(labels, dtype=torch.float32, device=device)) + model.fusion_regularization()
                if not torch.isfinite(loss):
                    raise RuntimeError("NONFINITE_LOSS")
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
        stop_scores = []
        for stage, view in views.items():
            ids = [r for r in stop if r in indexes[stage]]
            stop_scores.append(average_precision_score(view.target[[indexes[stage][r] for r in ids]], predict(stage, ids)))
        score = float(np.mean(stop_scores))
        if score > best_score:
            best_score, best_epoch, stale = score, epoch + 1, 0
            state = copy.deepcopy({k: v.detach().cpu() for k, v in model.state_dict().items()})
        else:
            stale += 1
        if stale >= patience:
            break
    model.load_state_dict(state)
    metrics, diagnostics = {}, []
    for stage, view in views.items():
        stop_ids = [r for r in stop if r in indexes[stage]]; valid_ids = [r for r in valid if r in indexes[stage]]
        train_ids = [r for r in fit if r in indexes[stage]]
        stop_score, valid_score = predict(stage, stop_ids), predict(stage, valid_ids)
        threshold_result = stage_threshold_metrics(view.target[[indexes[stage][r] for r in stop_ids]], stop_score,
                                                   view.target[[indexes[stage][r] for r in valid_ids]], valid_score)
        selected = threshold_result["stop_selected"]
        metrics[stage] = {key: float(selected[key]) for key in ("pr_auc", "roc_auc", "accuracy", "risk_precision", "risk_recall", "risk_f1")}
        metrics[stage]["selected_threshold"] = float(threshold_result["selected_threshold"])
        metrics[stage]["threshold_source"] = "stop_only"
        train_pr_auc = average_precision_score(view.target[[indexes[stage][r] for r in train_ids]], predict(stage, train_ids))
        metrics[stage]["train_pr_auc"] = float(train_pr_auc)
        metrics[stage]["generalization_gap"] = float(train_pr_auc - selected["pr_auc"])
        if domain == "oulad":
            outcomes = oulad_outcomes()
            labels = np.asarray([outcomes[r] for r in valid_ids])
            predicted = valid_score >= threshold_result["selected_threshold"]
            metrics[stage]["recall_fail"] = float(predicted[labels == "Fail"].mean()) if np.any(labels == "Fail") else None
            metrics[stage]["recall_withdrawn"] = float(predicted[labels == "Withdrawn"].mean()) if np.any(labels == "Withdrawn") else None
            non_withdrawn = labels != "Withdrawn"
            metrics[stage]["non_withdrawn_risk_recall"] = float(predicted[non_withdrawn][view.target[[indexes[stage][r] for r in valid_ids]][non_withdrawn] == 1].mean()) if np.any(non_withdrawn & (view.target[[indexes[stage][r] for r in valid_ids]] == 1)) else None
        # Re-run valid forward once for gate diagnostics, never for selection.
        model.eval(); forward(stage, valid_ids)
        if "gate_weights" in model.last_diagnostics:
            weights = model.last_diagnostics["gate_weights"].float().cpu().numpy(); entropy = model.last_diagnostics["gate_entropy"].float().cpu().numpy()
            diagnostics.append({"stage": stage, "tabular_mean": float(weights[:, 0].mean()), "tabular_std": float(weights[:, 0].std()),
                                "cnn_mean": float(weights[:, 1].mean()), "cnn_std": float(weights[:, 1].std()), "bilstm_mean": float(weights[:, 2].mean()), "bilstm_std": float(weights[:, 2].std()),
                                "gate_entropy_mean": float(entropy.mean()), "gate_entropy_std": float(entropy.std()), "temporal_utilization": float((weights[:, 1] + weights[:, 2]).mean())})
        else:
            alpha = model.last_diagnostics["alpha"].float().cpu().numpy()
            diagnostics.append({"stage": stage, "alpha_mean": float(alpha.mean()), "alpha_std": float(alpha.std()),
                                "temporal_delta_abs_mean": float(model.last_diagnostics["temporal_delta"].abs().float().mean().cpu())})
    result = {"scope": "single-fold single-seed screening; outer test unused", "domain": domain, "protocol": protocol,
            "temporal_variant": temporal_variant, "fold": fold, "seed": seed, "best_epoch": best_epoch,
            "fusion": fusion, "entropy_floor_coefficient": entropy_floor_coefficient, "branch_mode": branch_mode,
            "model_family": model_family, "residual_sample_gate": residual_sample_gate,
            "data_variant": data_variant,
            "include_final100": include_final100,
            "final_only": final_only,
            "stop_macro_pr_auc": best_score, "runtime_seconds": time.monotonic() - started,
            "parameter_count": sum(p.numel() for p in model.parameters()), "metrics": metrics,
            "gate_diagnostics": diagnostics, "outer_test_used": False,
            "final100_audit": final_audit if include_final100 else None}
    if collect_forensics:
        forensic = {"same_checkpoint_ablation": {}, "probes": {}, "gradients": {}, "valid_predictions": {}}
        masks = {"full": (True, True, True), "no_cnn": (True, False, True), "no_bilstm": (True, True, False), "no_temporal": (True, False, False)}
        for stage, view in views.items():
            train_ids = [r for r in fit if r in indexes[stage]]; valid_ids = [r for r in valid if r in indexes[stage]]
            target = view.target[[indexes[stage][r] for r in valid_ids]]
            scores = {name: predict(stage, valid_ids, mask) for name, mask in masks.items()}
            forensic["same_checkpoint_ablation"][stage] = {name: float(average_precision_score(target, score)) for name, score in scores.items()}
            forensic["valid_predictions"][stage] = {"record_id": valid_ids, "target": target.astype(int).tolist(), **{name: value.tolist() for name, value in scores.items()}}
            def reps(record_ids):
                model.eval()
                chunks = []
                for start in range(0, len(record_ids), batch_size):
                    inputs = batch_inputs(stage, record_ids[start:start + batch_size]); hs, hc, hl, ha = model.representations(*inputs[:5])
                    tabular = hs + ha * inputs[5].unsqueeze(-1)
                    chunks.append((tabular.detach().cpu().numpy(), hc.detach().cpu().numpy(), hl.detach().cpu().numpy()))
                return tuple(np.concatenate([chunk[i] for chunk in chunks]) for i in range(3))
            train_tab, train_cnn, train_lstm = reps(train_ids); valid_tab, valid_cnn, valid_lstm = reps(valid_ids)
            probe_inputs = {"tabular": (train_tab, valid_tab), "cnn": (train_cnn, valid_cnn), "bilstm": (train_lstm, valid_lstm),
                            "cnn_bilstm": (np.c_[train_cnn, train_lstm], np.c_[valid_cnn, valid_lstm]),
                            "all": (np.c_[train_tab, train_cnn, train_lstm], np.c_[valid_tab, valid_cnn, valid_lstm])}
            probe_result = {}
            y_train = view.target[[indexes[stage][r] for r in train_ids]]
            for name, (x_train, x_valid) in probe_inputs.items():
                scaler = StandardScaler().fit(x_train); probe = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=seed).fit(scaler.transform(x_train), y_train)
                probe_result[name] = float(average_precision_score(target, probe.predict_proba(scaler.transform(x_valid))[:, 1]))
            forensic["probes"][stage] = probe_result
            model.train(); model.zero_grad(set_to_none=True); chunk = train_ids[:min(batch_size, len(train_ids))]; logits, labels = forward(stage, chunk)
            loss_fn(logits, torch.tensor(labels, dtype=torch.float32, device=device)).backward()
            groups = {"tabular": ("static_projector", "aggregate_projector"), "cnn": ("cnn", "cnn_projection", "cnn_out"), "bilstm": ("bilstm", "lstm_out"), "gate": ("gate",), "head": ("head",)}
            gradients = {}
            for name, prefixes in groups.items():
                values = [parameter.grad.norm().item() for parameter_name, parameter in model.named_parameters() if parameter.grad is not None and parameter_name.startswith(prefixes)]
                gradients[name] = float(np.sqrt(np.sum(np.square(values)))) if values else 0.0
            forensic["gradients"][stage] = gradients
        result["forensics"] = forensic
    return result
