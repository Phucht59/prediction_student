"""Score frozen Hybrid, branch knockouts, and sklearn baselines. No HPO."""
from __future__ import annotations

import numpy as np
import torch
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from experiments.imbalance.samplers import PackedBatch, pack_features
from experiments.imbalance.train_hybrid import _batch_tensors, predict_scores
from src.prediction.model import Hybrid


def predict_hybrid(model: Hybrid, batch: PackedBatch, *, batch_size: int) -> np.ndarray:
    model.eval()
    device = next(model.parameters()).device
    return predict_scores(model.to(device), batch, batch_size=batch_size)


@torch.no_grad()
def predict_branch(model: Hybrid, batch: PackedBatch, branch: str, *, batch_size: int) -> np.ndarray:
    """Score a trained Hybrid using one representation only. Not a retrained ablation."""
    device = next(model.parameters()).device
    model.eval()
    out = []
    n = len(batch.target)
    for start in range(0, n, batch_size):
        idx = np.arange(start, min(n, start + batch_size))
        static, temporal, mask, lengths, aggregate, agg_ok, progress = _batch_tensors(batch, idx, device)
        if branch == "tabular":
            lengths = torch.zeros_like(lengths)
            mask = torch.zeros_like(mask)
            temporal = temporal * 0
        h_tab, h_cnn, h_lstm, temporal_available = model.representations(
            static, temporal, mask, lengths, aggregate, agg_ok
        )
        if branch == "tabular":
            fused = h_tab
        elif branch == "cnn":
            fused = torch.where(temporal_available.unsqueeze(-1), h_cnn, h_tab)
        elif branch == "bilstm":
            fused = torch.where(temporal_available.unsqueeze(-1), h_lstm, h_tab)
        else:
            raise ValueError(branch)
        logits = model.head(model.fusion_norm(fused)).squeeze(-1)
        out.append(torch.sigmoid(logits.float()).cpu().numpy())
    return np.concatenate(out).astype(np.float32)


@torch.no_grad()
def gate_masses(model: Hybrid, batch: PackedBatch, *, batch_size: int, cap: int = 4096) -> dict[str, float]:
    device = next(model.parameters()).device
    model.eval()
    masses = []
    n = min(len(batch.target), cap)
    for start in range(0, n, batch_size):
        idx = np.arange(start, min(n, start + batch_size))
        _ = model(*_batch_tensors(batch, idx, device))
        masses.append(model.last_diagnostics["gate_weights"].cpu().numpy())
    w = np.concatenate(masses, axis=0)
    return {
        "tabular_mass_mean": float(w[:, 0].mean()),
        "cnn_mass_mean": float(w[:, 1].mean()),
        "bilstm_mass_mean": float(w[:, 2].mean()),
        "n": int(len(w)),
    }


def fit_sklearn_baselines(train: PackedBatch, valid: PackedBatch, *, seed: int) -> dict[str, np.ndarray]:
    x_train = pack_features(train)
    y_train = np.asarray(train.target, dtype=int)
    x_valid = pack_features(valid)
    lr = LogisticRegression(class_weight="balanced", max_iter=1000, C=1.0, random_state=seed)
    rf = RandomForestClassifier(
        n_estimators=200, min_samples_leaf=2, class_weight="balanced", random_state=seed, n_jobs=-1
    )
    lr.fit(x_train, y_train)
    rf.fit(x_train, y_train)
    return {
        "LR": lr.predict_proba(x_valid)[:, 1].astype(np.float32),
        "RF": rf.predict_proba(x_valid)[:, 1].astype(np.float32),
        "models": {"LR": lr, "RF": rf},
    }
