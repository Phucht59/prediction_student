"""Locked splits + FIT-only scaling. Outer fold 0 is firewall only."""
from __future__ import annotations

import hashlib
import json
import shutil
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

from src.prediction.data.common import UnifiedHybridData
from src.prediction.data.preprocessing import ContextPreprocessor, MaskedStandardScaler
from src.prediction.data.uci import (
    UCI_CATEGORICAL_CONTEXT,
    UCI_NUMERIC_CONTEXT,
    build_uci_combined,
    build_uci_stage_view,
)
from src.prediction.data.oulad_features import OULAD_CATEGORICAL_CONTEXT, OULAD_NUMERIC_CONTEXT

from .paths import EXPECTED_SPLIT, KLTN_SPLITS, PHASE2_CACHE, RAW, SPLIT, ensure

DEVELOPMENT_OUTER = 0
UCI_STAGES = ("S0", "S1", "S2")
OULAD_STAGES = ("20pct", "35pct", "50pct", "75pct", "100pct")
UCI_PROGRESS = {"S0": 0.0, "S1": 0.5, "S2": 1.0}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def copy_locked_splits() -> dict[str, str]:
    ensure()
    observed = {}
    for name, expected in EXPECTED_SPLIT.items():
        src = KLTN_SPLITS / f"{name}.parquet"
        dst = SPLIT / f"{name}.parquet"
        if not src.exists():
            raise FileNotFoundError(f"LOCKED_SPLIT_MISSING:{src}")
        shutil.copy2(src, dst)
        digest = sha256_file(dst)
        if digest != expected:
            raise RuntimeError(f"SPLIT_HASH_MISMATCH:{name}:{digest}:{expected}")
        observed[name] = digest
    for extra in ("uci_outer.parquet", "oulad_outer.parquet"):
        src = KLTN_SPLITS / extra
        if src.exists():
            shutil.copy2(src, SPLIT / extra)
    return observed


def _group_col(frame: pd.DataFrame) -> str:
    if "group_id" in frame.columns:
        return "group_id"
    return "global_student_group"


def outer_test_ids(domain: str) -> set[str]:
    path = SPLIT / f"{domain}_outer.parquet"
    frame = pd.read_parquet(path)
    return set(frame.loc[frame.outer_fold == DEVELOPMENT_OUTER, "record_id"].astype(str))


def assert_no_outer(ids, domain: str) -> None:
    blocked = set(map(str, ids)) & outer_test_ids(domain)
    if blocked:
        raise RuntimeError(f"OUTER_FIREWALL_VIOLATION:{domain}:{len(blocked)}")


def inner_partitions(domain: str, context: pd.DataFrame, inner_fold: int) -> tuple[list[str], list[str], list[str]]:
    inner = pd.read_parquet(SPLIT / f"{domain}_inner.parquet")
    inner = inner[inner.outer_fold == DEVELOPMENT_OUTER].copy()
    inner["record_id"] = inner.record_id.astype(str)
    available = set(context.record_id.astype(str))
    valid_ids = set(inner.loc[inner.inner_fold == inner_fold, "record_id"]) & available
    train_ids = set(inner.loc[inner.inner_fold != inner_fold, "record_id"]) & available
    if train_ids & valid_ids:
        raise RuntimeError("INNER_TRAIN_VALID_OVERLAP")
    assert_no_outer(train_ids | valid_ids, domain)
    rest = context[context.record_id.astype(str).isin(train_ids)].drop_duplicates("record_id").reset_index(drop=True)
    splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    y = rest.target.to_numpy()
    groups = rest[_group_col(rest)].astype(str).to_numpy()
    fit_ids = stop_ids = None
    for fit, stop in splitter.split(rest, y, groups):
        if len(np.unique(y[fit])) == 2 and len(np.unique(y[stop])) == 2 and not (set(groups[fit]) & set(groups[stop])):
            fit_ids = rest.iloc[fit].record_id.astype(str).tolist()
            stop_ids = rest.iloc[stop].record_id.astype(str).tolist()
            break
    if fit_ids is None:
        raise RuntimeError("NO_FEASIBLE_FIT_STOP")
    valid = sorted(valid_ids)
    if set(fit_ids) & set(stop_ids) or set(fit_ids) & set(valid) or set(stop_ids) & set(valid):
        raise RuntimeError("FIT_STOP_VALID_OVERLAP")
    assert_no_outer(set(fit_ids) | set(stop_ids) | set(valid), domain)
    return fit_ids, stop_ids, valid


def _load_npz_view(stage_dir: Path) -> UnifiedHybridData:
    data = np.load(stage_dir / "view.npz", allow_pickle=False)
    metadata = json.loads((stage_dir / "metadata.json").read_text(encoding="utf-8"))
    n = len(data["record_id"])
    view = UnifiedHybridData(
        static=np.zeros((n, 0), np.float32),
        temporal=data["temporal"],
        temporal_mask=data["temporal_mask"].astype(bool),
        lengths=data["lengths"],
        aggregate=data["aggregate"],
        aggregate_available=data["aggregate_available"],
        progress=data["progress"],
        target=data["target"],
        record_id=data["record_id"].astype(str),
        group_id=data["group_id"].astype(str),
        metadata=metadata,
    )
    view.validate()
    return view


def _pad_views(views: dict[str, UnifiedHybridData]) -> dict[str, UnifiedHybridData]:
    max_t = max(view.temporal.shape[1] for view in views.values())
    padded = {}
    for stage, view in views.items():
        pad = max_t - view.temporal.shape[1]
        if pad <= 0:
            padded[stage] = view
            continue
        padded[stage] = UnifiedHybridData(
            static=view.static,
            temporal=np.pad(view.temporal, ((0, 0), (0, pad), (0, 0))),
            temporal_mask=np.pad(view.temporal_mask, ((0, 0), (0, pad))),
            lengths=view.lengths,
            aggregate=view.aggregate,
            aggregate_available=view.aggregate_available,
            progress=view.progress,
            target=view.target,
            record_id=view.record_id,
            group_id=view.group_id,
            metadata=view.metadata,
        )
        padded[stage].validate()
    return padded


def load_phase2_cache(domain: str) -> tuple[dict[str, UnifiedHybridData], pd.DataFrame]:
    root = PHASE2_CACHE / domain
    stages = UCI_STAGES if domain == "uci" else OULAD_STAGES
    views = _pad_views({stage: _load_npz_view(root / stage) for stage in stages})
    context = pd.read_parquet(root / "context.parquet")
    context["record_id"] = context.record_id.astype(str)
    if "group_id" not in context.columns and "global_student_group" in context.columns:
        context["group_id"] = context.global_student_group.astype(str)
    else:
        context["group_id"] = context.group_id.astype(str)
    return views, context


def build_uci_grade_views(mode: str) -> tuple[dict[str, UnifiedHybridData], pd.DataFrame]:
    """mode: both | temporal_only | aggregate_only."""
    frame, _ = build_uci_combined(RAW / "student-mat.csv", RAW / "student-por.csv")
    n = len(frame)
    g1 = frame.G1.to_numpy(np.float32) / 20.0
    g2 = frame.G2.to_numpy(np.float32) / 20.0
    views = {}
    for stage in UCI_STAGES:
        if mode == "both":
            views[stage] = build_uci_stage_view(frame, stage)
            continue
        temporal = np.zeros((n, 2, 1), np.float32)
        mask = np.zeros((n, 2), bool)
        aggregate = np.zeros((n, 5), np.float32)
        available = np.zeros(n, np.int8)
        if mode == "temporal_only":
            if stage in {"S1", "S2"}:
                temporal[:, 0, 0], mask[:, 0] = g1, True
            if stage == "S2":
                temporal[:, 1, 0], mask[:, 1] = g2, True
        elif mode == "aggregate_only":
            if stage == "S1":
                aggregate[:, 0], aggregate[:, 1], aggregate[:, 2], available[:] = g1, g1, 0.5, 1
            if stage == "S2":
                aggregate[:, 0] = g2
                aggregate[:, 1] = (g1 + g2) / 2
                aggregate[:, 2] = 1.0
                aggregate[:, 3] = g2 - g1
                aggregate[:, 4] = 1.0
                available[:] = 1
        else:
            raise ValueError(mode)
        views[stage] = UnifiedHybridData(
            static=np.zeros((n, 0), np.float32),
            temporal=temporal,
            temporal_mask=mask,
            lengths=mask.sum(1).astype(np.int64),
            aggregate=aggregate,
            aggregate_available=available,
            progress=np.full(n, UCI_PROGRESS[stage], np.float32),
            target=frame.target.to_numpy(np.int64),
            record_id=frame.record_id.to_numpy(str),
            group_id=frame.global_student_group.to_numpy(str),
            metadata={"dataset": "uci_combined", "stage": stage, "grade_mode": mode},
        )
        views[stage].validate()
    context = frame.rename(columns={"global_student_group": "group_id"})[
        ["record_id", "group_id", "target", "G1", "G2", *UCI_NUMERIC_CONTEXT, *UCI_CATEGORICAL_CONTEXT]
    ].copy()
    context["record_id"] = context.record_id.astype(str)
    context["group_id"] = context.group_id.astype(str)
    return views, context


@dataclass
class PreparedDomain:
    domain: str
    stages: tuple[str, ...]
    views: dict[str, UnifiedHybridData]
    context: pd.DataFrame
    static_map: dict[str, np.ndarray]
    static_dim: int
    temporal_dim: int
    aggregate_dim: int
    numeric: list[str]
    categorical: list[str]


def scale_prepared(domain: str, views: dict, context: pd.DataFrame, fit_ids: list[str]) -> PreparedDomain:
    numeric = list(UCI_NUMERIC_CONTEXT if domain == "uci" else OULAD_NUMERIC_CONTEXT)
    categorical = list(UCI_CATEGORICAL_CONTEXT if domain == "uci" else OULAD_CATEGORICAL_CONTEXT)
    views = {k: UnifiedHybridData(
        static=v.static.copy(),
        temporal=v.temporal.copy(),
        temporal_mask=v.temporal_mask.copy(),
        lengths=v.lengths.copy(),
        aggregate=v.aggregate.copy(),
        aggregate_available=v.aggregate_available.copy(),
        progress=v.progress.copy(),
        target=v.target.copy(),
        record_id=v.record_id.copy(),
        group_id=v.group_id.copy(),
        metadata=dict(v.metadata),
    ) for k, v in views.items()}
    fit_frame = context[context.record_id.astype(str).isin(fit_ids)].drop_duplicates("record_id")
    prep = ContextPreprocessor(numeric, categorical).fit(fit_frame)
    raw = prep.transform(context)
    static_map = {str(rid): raw[i] for i, rid in enumerate(context.record_id.astype(str))}
    aggs = []
    for view in views.values():
        lookup = {str(r): i for i, r in enumerate(view.record_id)}
        idx = [lookup[r] for r in fit_ids if r in lookup and view.aggregate_available[lookup[r]]]
        if idx:
            aggs.append(view.aggregate[idx])
    if aggs:
        stacked = np.concatenate(aggs)
        mean, std = stacked.mean(0), stacked.std(0)
        std = np.where(std < 1e-6, 1.0, std)
    else:
        mean = np.zeros(next(iter(views.values())).aggregate.shape[1], np.float32)
        std = np.ones_like(mean)
    for view in views.values():
        view.aggregate[:] = ((view.aggregate - mean) / std).astype(np.float32)
    xs, ms = [], []
    for view in views.values():
        lookup = {str(r): i for i, r in enumerate(view.record_id)}
        idx = [lookup[r] for r in fit_ids if r in lookup]
        if idx:
            xs.append(view.temporal[idx])
            ms.append(view.temporal_mask[idx])
    if xs:
        scaler = MaskedStandardScaler().fit(np.concatenate(xs), np.concatenate(ms))
        for view in views.values():
            view.temporal[:] = scaler.transform(view.temporal, view.temporal_mask)
    first = next(iter(views.values()))
    return PreparedDomain(
        domain=domain,
        stages=tuple(views),
        views=views,
        context=context,
        static_map=static_map,
        static_dim=int(prep.output_dim),
        temporal_dim=int(first.temporal.shape[2]),
        aggregate_dim=int(first.aggregate.shape[1]),
        numeric=numeric,
        categorical=categorical,
    )


def ids_for_stage(view, ids: list[str]) -> list[str]:
    present = set(map(str, view.record_id))
    return [i for i in ids if i in present]
