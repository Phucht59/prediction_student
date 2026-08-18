"""Reconstruct the three Phase8 Hybrid fitted instances without HPO/outer use.

This worker intentionally imports the preserved Phase8 authority implementation
from the backup ref and writes only reconstructed artifacts under this workspace.
It uses the frozen inner split bundle, P1 stage-balanced sampling, D3/F3, and
the fixed source defaults.  It does not inspect outer labels or rerun outer jobs.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import os
import random
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score


ROOT = Path(__file__).resolve().parents[2]
SOURCE_REPO = Path(r"C:\hufit\kltn")
AUTHORITY_REF = "codex/backup-hybrid-phase8-2026-08-17"
OUT = ROOT / "artifacts" / "prediction" / "reconstructed"
SEEDS = (42, 1201, 2026)
FOLDS = (0, 1, 2)
OOF_SEED_BY_FOLD = dict(zip(FOLDS, SEEDS, strict=True))
BATCH_SIZE = 256
MAX_EPOCHS = 16
PATIENCE = 5
GRADIENT_CLIP_NORM = 1.0
LEARNING_RATE = 8e-4
WEIGHT_DECAY = 2e-4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def authority_modules():
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    if str(SOURCE_REPO) not in sys.path:
        sys.path.insert(1, str(SOURCE_REPO))
    import scripts.audit.final_phase8_restore_acceptance as audit

    authority = audit.configure_authority_namespace()
    from src.hybrid.data.oulad import (
        OULAD_CATEGORICAL_CONTEXT,
        OULAD_NUMERIC_CONTEXT,
        build_compact_vle_daily,
        load_oulad_static_tables,
    )
    from src.hybrid.phase7 import execution as phase7_execution
    from src.hybrid.phase8.data_variants import apply_data_variant
    from src.hybrid.phase8.final100 import FINAL_STAGE, build_oulad_final100_view
    from src.hybrid.training.data import sample_prefixes_stage_balanced
    from src.hybrid.optimization.phase6b import stage_threshold_metrics
    return {
        "authority": authority,
        "phase7_execution": phase7_execution,
        "apply_data_variant": apply_data_variant,
        "FINAL_STAGE": FINAL_STAGE,
        "build_final100": build_oulad_final100_view,
        "load_oulad_static_tables": load_oulad_static_tables,
        "build_compact_vle_daily": build_compact_vle_daily,
        "oulad_numeric": OULAD_NUMERIC_CONTEXT,
        "oulad_categorical": OULAD_CATEGORICAL_CONTEXT,
        "sample_prefixes_stage_balanced": sample_prefixes_stage_balanced,
        "stage_threshold_metrics": stage_threshold_metrics,
    }


def load_views(instance: str, modules: dict[str, Any]):
    phase7_execution = modules["phase7_execution"]
    phase7_execution.ROOT = SOURCE_REPO
    if instance == "uci":
        views, context, numeric, categorical = phase7_execution.phase7_domain("uci")
        return views, context, numeric, categorical, {"data_variant": "D0_raw", "endpoint_scope": ["S0", "S1", "S2"]}
    if instance == "oulad_early":
        views, context, numeric, categorical = phase7_execution.phase7_domain("oulad")
        views = {stage: modules["apply_data_variant"](view, "D3_both_safe") for stage, view in views.items()}
        return views, context, numeric, categorical, {"data_variant": "D3_both_safe", "endpoint_scope": ["20pct", "35pct", "50pct", "75pct"]}
    if instance != "oulad_final":
        raise ValueError(f"unknown instance: {instance}")
    raw = SOURCE_REPO / "data" / "raw"
    runtime = SOURCE_REPO / "artifacts" / "hybrid" / "phase1" / "runtime"
    _, _, base = modules["load_oulad_static_tables"](raw)
    daily = modules["build_compact_vle_daily"](raw, runtime)
    base, view, audit = modules["build_final100"](base, daily, raw)
    view = modules["apply_data_variant"](view, "D3_both_safe")
    context_columns = ["record_id", "group_id", "target", *modules["oulad_numeric"], *modules["oulad_categorical"]]
    context = base.loc[:, [column for column in context_columns if column in base.columns]].drop_duplicates("record_id").reset_index(drop=True)
    return {modules["FINAL_STAGE"]: view}, context, modules["oulad_numeric"], modules["oulad_categorical"], {"data_variant": "D3_both_safe", "endpoint_scope": [modules["FINAL_STAGE"]], "final100_audit": audit}


def partitions(instance: str, context: pd.DataFrame, phase7_execution):
    domain = "uci" if instance == "uci" else "oulad"
    return phase7_execution._partitions(domain, 0, context)


def prepare_views(views, context, numeric, categorical, fit_ids, phase7_execution):
    local_views = copy.deepcopy(views)
    static_map, preprocessor = phase7_execution._scale(local_views, context, numeric, categorical, fit_ids, "uci" if len(numeric) == 12 else "oulad")
    return local_views, static_map, preprocessor


def train_state(*, instance: str, views, context: pd.DataFrame, numeric, categorical, fit_ids: list[str], stop_ids: list[str], valid_ids: list[str], seed: int, epochs: int | None, modules: dict[str, Any], collect_validation: bool):
    authority = modules["authority"]
    phase7_execution = modules["phase7_execution"]
    local_views, static_map, preprocessor = prepare_views(views, context, numeric, categorical, fit_ids, phase7_execution)
    domain = "uci" if instance == "uci" else "oulad"
    indices = {stage: {record_id: i for i, record_id in enumerate(view.record_id.astype(str))} for stage, view in local_views.items()}
    stages = tuple(local_views)
    first = next(iter(local_views.values()))
    config = authority["config"](
        int(preprocessor.output_dim),
        int(first.temporal.shape[2]),
        int(first.aggregate.shape[1]),
        fusion="adaptive_entropy",
        entropy_floor_coefficient=0.002,
        branch_mode="full",
    )
    model = authority["model"](config).to(DEVICE)
    seed_everything(seed)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    fit_targets = np.asarray([int(context.loc[context.record_id.astype(str) == record_id, "target"].iloc[0]) for record_id in fit_ids])
    pos_weight = torch.tensor([(len(fit_targets) - fit_targets.sum()) / max(1, fit_targets.sum())], dtype=torch.float32, device=DEVICE)
    loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    def batch_inputs(stage: str, record_ids: list[str]):
        view = local_views[stage]
        ix = np.asarray([indices[stage][record_id] for record_id in record_ids])
        static = np.asarray([static_map[record_id] for record_id in record_ids], dtype=np.float32)
        return (
            torch.tensor(static, dtype=torch.float32, device=DEVICE),
            torch.tensor(view.temporal[ix], dtype=torch.float32, device=DEVICE),
            torch.tensor(view.temporal_mask[ix], dtype=torch.bool, device=DEVICE),
            torch.tensor(view.lengths[ix], dtype=torch.long, device=DEVICE),
            torch.tensor(view.aggregate[ix], dtype=torch.float32, device=DEVICE),
            torch.tensor(view.aggregate_available[ix], dtype=torch.float32, device=DEVICE),
            torch.tensor(view.progress[ix], dtype=torch.float32, device=DEVICE),
        )

    def forward(stage: str, record_ids: list[str]):
        logits = model(*batch_inputs(stage, record_ids))
        view = local_views[stage]
        labels = view.target[np.asarray([indices[stage][record_id] for record_id in record_ids])]
        return logits, labels

    @torch.no_grad()
    def predict(stage: str, record_ids: list[str]) -> np.ndarray:
        if not record_ids:
            return np.empty(0, dtype=np.float32)
        model.eval()
        outputs = []
        for start in range(0, len(record_ids), BATCH_SIZE):
            logits, _ = forward(stage, record_ids[start : start + BATCH_SIZE])
            outputs.append(torch.sigmoid(logits).detach().cpu().numpy())
        return np.concatenate(outputs).astype(np.float32)

    available = {record_id: [stage for stage in stages if record_id in indices[stage]] for record_id in fit_ids}
    best_score = -np.inf
    best_epoch = 0
    stale = 0
    best_state = None
    target_epochs = int(epochs or MAX_EPOCHS)
    for epoch in range(target_epochs):
        model.train()
        choices = modules["sample_prefixes_stage_balanced"](fit_ids, [available[record_id] for record_id in fit_ids], seed, epoch)
        by_stage = {stage: [] for stage in stages}
        for record_id, stage in zip(fit_ids, choices, strict=True):
            by_stage[stage].append(record_id)
        for stage, selected_ids in by_stage.items():
            for start in range(0, len(selected_ids), BATCH_SIZE):
                chunk = selected_ids[start : start + BATCH_SIZE]
                if not chunk:
                    continue
                optimizer.zero_grad(set_to_none=True)
                logits, labels = forward(stage, chunk)
                loss = loss_fn(logits, torch.tensor(labels, dtype=torch.float32, device=DEVICE)) + model.fusion_regularization()
                if not torch.isfinite(loss):
                    raise RuntimeError(f"NONFINITE_LOSS:{instance}:{stage}:{epoch}")
                loss.backward()
                norm = torch.nn.utils.clip_grad_norm_(model.parameters(), GRADIENT_CLIP_NORM)
                if not torch.isfinite(norm):
                    raise RuntimeError(f"NONFINITE_GRADIENT:{instance}:{stage}:{epoch}")
                optimizer.step()
        if not stop_ids:
            continue
        stop_scores = []
        for stage, view in local_views.items():
            stage_ids = [record_id for record_id in stop_ids if record_id in indices[stage]]
            if not stage_ids:
                continue
            stage_indices = [indices[stage][record_id] for record_id in stage_ids]
            stop_scores.append(average_precision_score(view.target[stage_indices], predict(stage, stage_ids)))
        score = float(np.mean(stop_scores)) if stop_scores else float("nan")
        if np.isfinite(score) and score > best_score:
            best_score = score
            best_epoch = epoch + 1
            stale = 0
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        else:
            stale += 1
        if stale >= PATIENCE:
            break
    if best_state is None:
        best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        best_epoch = target_epochs
    model.load_state_dict(best_state, strict=True)

    thresholds: dict[str, float] = {}
    validation_rows: list[dict[str, Any]] = []
    if collect_validation:
        for stage, view in local_views.items():
            stage_stop = [record_id for record_id in stop_ids if record_id in indices[stage]]
            stage_valid = [record_id for record_id in valid_ids if record_id in indices[stage]]
            stage_stop.sort()
            stage_valid.sort()
            stop_ix = [indices[stage][record_id] for record_id in stage_stop]
            valid_ix = [indices[stage][record_id] for record_id in stage_valid]
            stop_prob = predict(stage, stage_stop)
            valid_prob = predict(stage, stage_valid)
            if len(stop_ix) and len(valid_ix):
                threshold_result = modules["stage_threshold_metrics"](view.target[stop_ix], stop_prob, view.target[valid_ix], valid_prob)
                threshold = float(threshold_result["selected_threshold"])
            else:
                threshold = 0.5
            thresholds[stage] = threshold
            for record_id, probability in zip(stage_valid, valid_prob, strict=True):
                index = indices[stage][record_id]
                validation_rows.append({
                    "record_id": record_id,
                    "group_id": str(view.group_id[index]),
                    "stage": stage,
                    "target": int(view.target[index]),
                    "risk_probability": float(probability),
                    "predicted_risk": int(probability >= threshold),
                    "threshold": threshold,
                    "hybrid_uncertainty": float(-(probability * math.log(max(float(probability), 1e-12)) + (1.0 - probability) * math.log(max(float(1.0 - probability), 1e-12))) / math.log(2.0)),
                    "fold": None,
                    "seed": seed,
                    "model_id": "hybrid",
                })
    return {
        "state_dict": best_state,
        "config": config,
        "best_epoch": int(best_epoch),
        "best_stop_macro_pr_auc": None if not np.isfinite(best_score) else float(best_score),
        "thresholds": thresholds,
        "validation_rows": validation_rows,
        "parameter_count": int(sum(parameter.numel() for parameter in model.parameters())),
        "preprocessor_output_dim": int(preprocessor.output_dim),
        "fit_count": len(fit_ids),
        "stop_count": len(stop_ids),
    }


def save_active_checkpoint(instance: str, result: dict[str, Any], path: Path, metadata: dict[str, Any]) -> dict[str, Any]:
    from src.prediction.model import Hybrid, HybridConfig
    from src.prediction.training.checkpoints import save_checkpoint

    cfg = HybridConfig(**result["config"].__dict__)
    active = Hybrid(cfg)
    active.load_state_dict(result["state_dict"], strict=True)
    save_checkpoint(path, active, instance=instance, metadata=metadata)
    return {"path": str(path), "sha256": sha256_file(path), "parameter_count": int(sum(parameter.numel() for parameter in active.parameters())), "config_hash": sha256_bytes(json.dumps(cfg.__dict__, sort_keys=True, default=str).encode())}


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    modules = authority_modules()
    # Explicitly verify the replay source and the frozen split identity before training.
    model_selection = json.loads((ROOT / "artifacts" / "prediction" / "final" / "development" / "model_selection.json").read_text(encoding="utf-8"))
    split_paths = {"uci": SOURCE_REPO / "artifacts" / "hybrid" / "phase1" / "splits" / "uci_inner.parquet", "oulad": SOURCE_REPO / "artifacts" / "hybrid" / "phase1" / "splits" / "oulad_inner.parquet"}
    split_hashes = {name: sha256_file(path) for name, path in split_paths.items()}
    expected_split_hashes = {"uci": model_selection["split_identity"]["artifacts/hybrid/phase1/splits/uci_inner.parquet"], "oulad": model_selection["split_identity"]["artifacts/hybrid/phase1/splits/oulad_inner.parquet"]}
    if split_hashes != expected_split_hashes:
        raise RuntimeError(f"FROZEN_SPLIT_HASH_MISMATCH:{split_hashes}:{expected_split_hashes}")

    manifest: dict[str, Any] = {
        "status": "RUNNING",
        "reconstruction_identity": "RECONSTRUCTED_FROM_FROZEN_PROTOCOL",
        "model_id": "hybrid",
        "display_name": "Hybrid",
        "device": str(DEVICE),
        "hpo": False,
        "outer_test_used": False,
        "outer_reselection": False,
        "protocol": {"architecture": "Phase8 D3/F3 Hybrid", "data_variant": "D3_both_safe", "fusion": "F3_adaptive_entropy", "entropy_floor_coefficient": 0.002, "training_protocol": "P1_stage_balanced", "batch_size": BATCH_SIZE, "max_epochs": MAX_EPOCHS, "patience": PATIENCE, "gradient_clip_norm": GRADIENT_CLIP_NORM, "optimizer": "AdamW", "learning_rate": LEARNING_RATE, "weight_decay": WEIGHT_DECAY, "loss": "BCEWithLogitsLoss + F3 entropy regularization", "class_weight": "FIT-only positive weight", "seeds": list(SEEDS), "folds": list(FOLDS)},
        "split_hashes": split_hashes,
        "instances": {},
    }
    for instance in ("uci", "oulad_early", "oulad_final"):
        print(f"START {instance}", flush=True)
        views, context, numeric, categorical, data_meta = load_views(instance, modules)
        fit, stop, valid = partitions(instance, context, modules["phase7_execution"])
        fit_set = set(fit)
        stop_set = set(stop)
        valid_set = set(valid)
        if fit_set & stop_set or fit_set & valid_set or stop_set & valid_set:
            raise RuntimeError(f"SPLIT_OVERLAP:{instance}")
        instance_dir = OUT / instance
        oof_dir = instance_dir / "oof"
        instance_rows: list[dict[str, Any]] = []
        fold_results: list[dict[str, Any]] = []
        for fold in FOLDS:
            fold_fit, fold_stop, fold_valid = partitions(instance, context, modules["phase7_execution"])
            # _partitions is fixed at the requested fold; rerun with explicit fold below.
            domain = "uci" if instance == "uci" else "oulad"
            split = pd.read_parquet(split_paths[domain])
            split = split[split.outer_fold == 0].copy()
            val_ids = set(split.loc[split.inner_fold == fold, "record_id"].astype(str))
            train_pool = set(split.loc[split.inner_fold != fold, "record_id"].astype(str))
            fold_frame = context[context.record_id.astype(str).isin(train_pool)].drop_duplicates("record_id").reset_index(drop=True)
            fi, si = modules["phase7_execution"].split_fit_stop(fold_frame)
            fold_fit = fold_frame.iloc[fi].record_id.astype(str).tolist()
            fold_stop = fold_frame.iloc[si].record_id.astype(str).tolist()
            fold_valid = sorted(val_ids)
            if set(fold_fit) & set(fold_valid) or set(fold_stop) & set(fold_valid):
                raise RuntimeError(f"OOF_LEAKAGE:{instance}:fold={fold}")
            seed = OOF_SEED_BY_FOLD[fold]
            result = train_state(instance=instance, views=views, context=context, numeric=numeric, categorical=categorical, fit_ids=fold_fit, stop_ids=fold_stop, valid_ids=fold_valid, seed=seed, epochs=None, modules=modules, collect_validation=True)
            for row in result["validation_rows"]:
                row["fold"] = fold
                row["seed"] = seed
                instance_rows.append(row)
            checkpoint_meta = {"reconstruction_identity": "RECONSTRUCTED_FROM_FROZEN_PROTOCOL", "instance": instance, "fold": fold, "seed": seed, "outer_test_used": False, "split_hash": split_hashes[domain], "source_authority_ref": AUTHORITY_REF, "best_epoch": result["best_epoch"], "best_stop_macro_pr_auc": result["best_stop_macro_pr_auc"]}
            checkpoint = save_active_checkpoint(instance, result, oof_dir / f"fold_{fold}_seed_{seed}.pt", checkpoint_meta)
            fold_results.append({"fold": fold, "seed": seed, "best_epoch": result["best_epoch"], "best_stop_macro_pr_auc": result["best_stop_macro_pr_auc"], "checkpoint": checkpoint, "fit_count": result["fit_count"], "stop_count": result["stop_count"], "valid_count": len(fold_valid)})
            print(f"DONE {instance} fold={fold} seed={seed} epoch={result['best_epoch']}", flush=True)
        oof = pd.DataFrame(instance_rows)
        if oof.empty:
            raise RuntimeError(f"EMPTY_OOF:{instance}")
        if oof.duplicated(["record_id", "stage"]).any() or oof.group_id.isna().any():
            raise RuntimeError(f"OOF_IDENTITY_OR_GROUP_FAILURE:{instance}")
        oof.to_parquet(instance_dir / "oof_predictions.parquet", index=False)
        final_epoch = int(np.median([item["best_epoch"] for item in fold_results]))
        domain = "uci" if instance == "uci" else "oulad"
        split = pd.read_parquet(split_paths[domain])
        context_ids = set(context.record_id.astype(str))
        full_train_ids = [
            record_id
            for record_id in split.loc[split.outer_fold == 0, "record_id"].astype(str).drop_duplicates().tolist()
            if record_id in context_ids
        ]
        if not full_train_ids:
            raise RuntimeError(f"EMPTY_FINAL_FIT_CONTEXT:{instance}")
        final_result = train_state(instance=instance, views=views, context=context, numeric=numeric, categorical=categorical, fit_ids=full_train_ids, stop_ids=[], valid_ids=[], seed=42, epochs=final_epoch, modules=modules, collect_validation=False)
        threshold_by_stage = {}
        for stage in oof.stage.unique():
            threshold_by_stage[stage] = float(oof.loc[oof.stage == stage, "threshold"].median())
        final_meta = {"reconstruction_identity": "RECONSTRUCTED_FROM_FROZEN_PROTOCOL", "instance": instance, "seed": 42, "epoch_policy": "median_inner_fold_best_epoch", "epochs": final_epoch, "thresholds_from_inner_oof": threshold_by_stage, "outer_test_used": False, "split_hash": split_hashes[domain], "source_authority_ref": AUTHORITY_REF, "data_meta": data_meta}
        final_checkpoint = save_active_checkpoint(instance, final_result, instance_dir / "final_hybrid.pt", final_meta)
        write_json(instance_dir / "reconstruction_manifest.json", {"status": "COMPLETE", "reconstruction_identity": "RECONSTRUCTED_FROM_FROZEN_PROTOCOL", "instance": instance, "model_id": "hybrid", "parameter_count": final_result["parameter_count"], "config": final_result["config"].__dict__, "final_checkpoint": final_checkpoint, "oof_prediction_path": str(instance_dir / "oof_predictions.parquet"), "oof_rows": int(len(oof)), "oof_group_count": int(oof.group_id.nunique()), "fold_results": fold_results, "data_meta": data_meta, "thresholds_from_inner_oof": threshold_by_stage, "historical_outer_evidence_assignment": False})
        manifest["instances"][instance] = {"final_checkpoint": final_checkpoint, "oof_rows": int(len(oof)), "oof_group_count": int(oof.group_id.nunique()), "fold_results": fold_results, "final_epoch": final_epoch, "data_meta": data_meta}
        print(f"DONE {instance} final epoch={final_epoch} oof_rows={len(oof)}", flush=True)
    manifest["status"] = "COMPLETE"
    write_json(OUT / "reconstruction_manifest.json", manifest)
    print("RECONSTRUCTION_COMPLETE", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
