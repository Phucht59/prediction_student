"""C0_DOWNSTREAM_EVIDENCE_MATERIALIZATION.

Train frozen Phase4 Hybrid C0 on inner FIT/STOP only. Predict VALID OOF.
Does not change Phase4 canonical metrics or architecture.
"""
from __future__ import annotations

import json
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score

from experiments.hybrid_vnext.data import inner_partitions, load_domain_phase4, scale_views
from experiments.hybrid_vnext.metrics import select_stop_threshold
from experiments.hybrid_vnext.phase4_common import PHASE3_HPO, SHARED_STRUCTURAL
from experiments.hybrid_vnext.protocol import ROOT, require_cuda, seed_everything, sha256_file, verify_split_hashes
from src.prediction.model import Hybrid, HybridConfig

OUT = ROOT / "artifacts" / "recommend_hybrid" / "v3" / "data"
STAGES = ("20pct", "35pct", "50pct", "75pct", "100pct")
REC_STAGES = ("20pct", "35pct", "50pct", "75pct")
HP = PHASE3_HPO["oulad"]
KLTN_RAW = Path(r"C:\hufit\kltn\data\raw")


def _entropy(p: np.ndarray) -> np.ndarray:
    q = np.clip(p, 1e-12, 1 - 1e-12)
    return (-(q * np.log(q) + (1 - q) * np.log(1 - q)) / math.log(2.0)).astype(np.float32)


def _pr_auc(y, p) -> float:
    if len(np.unique(y)) < 2:
        return float("nan")
    return float(average_precision_score(y, p))


def _ids_for_stage(view, ids: list[str]) -> list[str]:
    present = set(map(str, view.record_id))
    return [i for i in ids if i in present]


class C0Materializer:
    def __init__(self, prepared, *, seed: int):
        require_cuda()
        self.prepared = prepared
        self.seed = seed
        self.device = torch.device("cuda")
        self.batch_size = int(HP["batch_size"])

    def _tensors(self, stage: str, ids: list[str]):
        view = self.prepared.views[stage]
        lookup = {str(r): i for i, r in enumerate(view.record_id)}
        idx = np.asarray([lookup[i] for i in ids])
        return {
            "static": torch.tensor(np.asarray([self.prepared.static_map[i] for i in ids]), dtype=torch.float32, device=self.device),
            "temporal": torch.tensor(view.temporal[idx], dtype=torch.float32, device=self.device),
            "temporal_mask": torch.tensor(view.temporal_mask[idx], device=self.device),
            "lengths": torch.tensor(view.lengths[idx], device=self.device),
            "aggregate": torch.tensor(view.aggregate[idx], dtype=torch.float32, device=self.device),
            "aggregate_available": torch.tensor(view.aggregate_available[idx], device=self.device),
            "progress": torch.tensor(view.progress[idx], dtype=torch.float32, device=self.device),
        }, view.target[idx].astype(np.float32)

    def predict(self, model: Hybrid, stage: str, ids: list[str]) -> np.ndarray:
        if not ids:
            return np.empty(0, np.float32)
        model.eval()
        out = []
        with torch.inference_mode():
            for start in range(0, len(ids), self.batch_size):
                chunk = ids[start : start + self.batch_size]
                inputs, _ = self._tensors(stage, chunk)
                logits = model(**inputs)
                out.append(torch.sigmoid(logits.float()).cpu().numpy())
        return np.concatenate(out).astype(np.float32)

    def fit(self, fit_ids: list[str], stop_ids: list[str]) -> dict:
        seed_everything(self.seed)
        cfg = HybridConfig(
            static_dim=self.prepared.static_dim,
            temporal_dim=self.prepared.temporal_dim,
            aggregate_dim=self.prepared.aggregate_dim,
            d_fuse=SHARED_STRUCTURAL["d_fuse"],
            cnn_channels=SHARED_STRUCTURAL["cnn_channels"],
            bilstm_hidden=SHARED_STRUCTURAL["bilstm_hidden"],
            dropout=float(HP["dropout"]),
            entropy_floor_coefficient=float(HP["entropy_floor_coefficient"]),
        )
        model = Hybrid(cfg).to(self.device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=float(HP["lr"]), weight_decay=float(HP["weight_decay"]))
        rec = self.prepared.context.record_id.astype(str).to_numpy()
        tgt = self.prepared.context["target"].to_numpy()
        id_to_y = {str(a): int(b) for a, b in zip(rec, tgt)}
        y = np.asarray([id_to_y[i] for i in fit_ids])
        pos_weight = torch.tensor([(len(y) - y.sum()) / max(1, y.sum()) * HP["pos_weight_multiplier"]], dtype=torch.float32, device=self.device)
        loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        scaler = torch.amp.GradScaler("cuda", enabled=True)
        best = -np.inf
        best_state = None
        stale = 0
        history = []
        started = time.monotonic()
        for epoch in range(1, 61):
            model.train()
            rng = np.random.default_rng(self.seed + epoch)
            epoch_loss = []
            for stage in STAGES:
                ids = _ids_for_stage(self.prepared.views[stage], fit_ids)
                rng.shuffle(ids)
                for start in range(0, len(ids), self.batch_size):
                    chunk = ids[start : start + self.batch_size]
                    if len(chunk) < 2:
                        continue
                    inputs, labels = self._tensors(stage, chunk)
                    optimizer.zero_grad(set_to_none=True)
                    with torch.amp.autocast("cuda", enabled=True):
                        logits = model(**inputs)
                        loss = loss_fn(logits, torch.tensor(labels, device=self.device)) + model.fusion_regularization()
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    scaler.step(optimizer)
                    scaler.update()
                    epoch_loss.append(float(loss.detach().cpu()))
            stop_scores = []
            for stage in STAGES:
                ids = _ids_for_stage(self.prepared.views[stage], stop_ids)
                if not ids:
                    continue
                pred = self.predict(model, stage, ids)
                lookup = {str(r): i for i, r in enumerate(self.prepared.views[stage].record_id)}
                y_stop = self.prepared.views[stage].target[[lookup[i] for i in ids]]
                stop_scores.append(_pr_auc(y_stop, pred))
            macro = float(np.nanmean(stop_scores)) if stop_scores else float("nan")
            history.append({"epoch": epoch, "loss": float(np.mean(epoch_loss) if epoch_loss else np.nan), "stop_macro_pr_auc": macro})
            print(f"epoch={epoch} stop_macro={macro:.4f}", flush=True)
            if macro > best:
                best = macro
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                stale = 0
            else:
                stale += 1
            if stale >= 10:
                break
        if best_state is None:
            raise RuntimeError("NO_C0_CHECKPOINT")
        model.load_state_dict(best_state)
        model.to(self.device)
        return {"model": model, "best_stop_macro_pr_auc": best, "history": history, "runtime_seconds": time.monotonic() - started, "parameter_count": sum(p.numel() for p in model.parameters())}


def materialize(folds=(0, 1, 2), seed: int = 42) -> pd.DataFrame:
    OUT.mkdir(parents=True, exist_ok=True)
    split_hashes = verify_split_hashes()
    views, context, numeric, categorical = load_domain_phase4("oulad")
    rows = []
    fold_reports = []
    from src.prediction.data.oulad import load_oulad_static_tables

    _, _, base = load_oulad_static_tables(KLTN_RAW)
    identity = base[["record_id", "id_student", "code_module", "code_presentation", "module_presentation_length", "date_registration"]].copy()
    identity["record_id"] = identity.record_id.astype(str)
    for fold in folds:
        fit, stop, valid = inner_partitions("oulad", context, fold)
        prepared = scale_views(views, context, numeric, categorical, fit, "oulad")
        trainer = C0Materializer(prepared, seed=seed)
        fitted = trainer.fit(fit, stop)
        ckpt = OUT / f"c0_inner_fold{fold}_seed{seed}.pt"
        torch.save({"model_id": "hybrid", "instance": "oulad", "fold": fold, "seed": seed, "state_dict": fitted["model"].state_dict(), "config": fitted["model"].config.__dict__}, ckpt)
        fold_row = {"fold": fold, "seed": seed, "best_stop_macro_pr_auc": fitted["best_stop_macro_pr_auc"], "parameter_count": fitted["parameter_count"], "checkpoint": str(ckpt)}
        for stage in REC_STAGES:
            ids = _ids_for_stage(prepared.views[stage], valid)
            pred = trainer.predict(fitted["model"], stage, ids)
            lookup = {str(r): i for i, r in enumerate(prepared.views[stage].record_id)}
            y = prepared.views[stage].target[[lookup[i] for i in ids]]
            threshold = float(select_stop_threshold(np.asarray([prepared.views[stage].target[lookup[i]] for i in _ids_for_stage(prepared.views[stage], stop)]), trainer.predict(fitted["model"], stage, _ids_for_stage(prepared.views[stage], stop)))) if _ids_for_stage(prepared.views[stage], stop) else 0.5
            fold_row[f"{stage}_valid_pr_auc"] = _pr_auc(y, pred)
            fold_row[f"{stage}_threshold"] = threshold
            unc = _entropy(pred)
            for record_id, p, u in zip(ids, pred, unc):
                rows.append(
                    {
                        "record_id": record_id,
                        "inner_fold": fold,
                        "seed": seed,
                        "stage_or_endpoint": stage,
                        "risk_probability": float(p),
                        "prediction_threshold": threshold,
                        "predicted_risk": int(p >= threshold),
                        "uncertainty": float(u),
                        "split_role": "VALID_OOF",
                    }
                )
        fold_reports.append(fold_row)
        print("FOLD_DONE", fold_row, flush=True)
    frame = pd.DataFrame(rows)
    ident = identity.drop_duplicates("record_id")
    frame = frame.merge(ident, on="record_id", how="left")
    if frame[["id_student", "code_module", "code_presentation"]].isna().any().any():
        raise RuntimeError("IDENTITY_JOIN_FAILED")
    frame["student_key"] = frame["id_student"].astype(str)
    frame["course_key"] = frame["code_module"].astype(str) + "::" + frame["code_presentation"].astype(str)
    frame["stage"] = frame["stage_or_endpoint"].map({"20pct": "EARLY_20", "35pct": "EARLY_35", "50pct": "MIDDLE_50", "75pct": "LATE_75"})
    frame["cutoff_day"] = np.maximum(1, np.floor(frame["module_presentation_length"] * frame["stage_or_endpoint"].map({"20pct": 0.2, "35pct": 0.35, "50pct": 0.5, "75pct": 0.75})).astype(int))
    frame["query_id"] = frame["student_key"] + "::" + frame["code_module"].astype(str) + "::" + frame["code_presentation"].astype(str) + "::" + frame["stage"]
    if frame["query_id"].duplicated().any():
        raise RuntimeError("DUPLICATE_QUERY")
    if (frame["stage_or_endpoint"] == "100pct").any():
        raise RuntimeError("HUNDRED_PCT_LEAKED")
    path = OUT / "c0_oof_predictions.parquet"
    frame.to_parquet(path, index=False)
    provenance = {
        "label": "C0_DOWNSTREAM_EVIDENCE_MATERIALIZATION",
        "prediction_authority": "Phase4 Hybrid C0",
        "architecture": SHARED_STRUCTURAL,
        "training": HP,
        "seeds": [seed],
        "inner_folds": list(folds),
        "split_hashes": split_hashes,
        "outer_test_used": False,
        "canonical_phase4_metrics_changed": False,
        "hpo_performed": False,
        "fold_reports": fold_reports,
        "row_count": int(len(frame)),
        "query_count": int(frame.query_id.nunique()),
        "uncertainty_formula": "binary_shannon_entropy_H2(p)",
    }
    (OUT / "C0_PREDICTION_PROVENANCE.json").write_text(json.dumps(provenance, indent=2) + "\n", encoding="utf-8")
    print("WROTE", path, len(frame))
    return frame


if __name__ == "__main__":
    materialize()
