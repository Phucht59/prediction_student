"""Cutoff-safe inner views, FIT-only scaling, and feature-parity frames."""
from __future__ import annotations

import copy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .protocol import (
    ART,
    CACHE,
    DEVELOPMENT_OUTER_FOLD,
    FORBIDDEN_OULAD,
    FORBIDDEN_UCI,
    KLTN,
    OULAD_PRIMARY,
    UCI_STAGES,
    assert_disjoint,
    assert_no_outer,
    bootstrap_kltn_namespace,
    sha256_text,
    split_paths,
    write_json,
)


@dataclass
class PreparedDomain:
    domain: str
    stages: tuple[str, ...]
    views: dict[str, Any]
    context: pd.DataFrame
    numeric: list[str]
    categorical: list[str]
    static_map: dict[str, np.ndarray]
    summary_map: dict[str, dict[str, np.ndarray]]
    static_dim: int
    temporal_dim: int
    aggregate_dim: int
    summary_dim: int
    feature_contract: dict[str, Any]


def _apply_d3(view):
    from src.hybrid.phase7.contracts import UnifiedHybridData
    from src.hybrid.phase7.data import OULAD_PHASE7_AGGREGATE_CHANNELS, OULAD_PHASE7_TEMPORAL_CHANNELS

    temporal_index = {name: i for i, name in enumerate(OULAD_PHASE7_TEMPORAL_CHANNELS)}
    aggregate_index = {name: i for i, name in enumerate(OULAD_PHASE7_AGGREGATE_CHANNELS)}
    count_channels = ("activity_intensity_log1p", "content_activity", "forum_activity", "quiz_activity", "assessment_related_activity")
    unique_channels = ("unique_sites", "unique_activity_types")
    temporal = view.temporal.copy().astype(np.float32)
    aggregate = view.aggregate.copy().astype(np.float32)
    raw = temporal.copy()
    intensity = temporal_index["activity_intensity_log1p"]
    raw[..., intensity] = np.expm1(np.clip(temporal[..., intensity], 0.0, 30.0))
    exposure = temporal[..., temporal_index["week_exposure_fraction"]]
    days = exposure * 7.0
    valid = view.temporal_mask & (days > 0)
    for name in count_channels:
        idx = temporal_index[name]
        rate = np.divide(raw[..., idx], days, out=np.zeros_like(raw[..., idx]), where=valid)
        temporal[..., idx] = np.log1p(np.maximum(rate, 0.0))
    idx = temporal_index["active_days"]
    temporal[..., idx] = np.divide(raw[..., idx], days, out=np.zeros_like(raw[..., idx]), where=valid)
    for name in unique_channels:
        idx = temporal_index[name]
        rate = np.divide(raw[..., idx], days, out=np.zeros_like(raw[..., idx]), where=valid)
        temporal[..., idx] = np.log1p(np.maximum(rate, 0.0))
    temporal[~view.temporal_mask] = 0.0
    activity = raw[..., intensity]
    total_days = days * valid
    daily = np.divide((activity * valid).sum(1), total_days.sum(1), out=np.zeros(len(view.record_id), np.float32), where=total_days.sum(1) > 0)
    last = np.maximum(view.temporal_mask.sum(1) - 1, 0)
    rows = np.arange(len(view.record_id))
    last_rate = np.divide(activity[rows, last], days[rows, last], out=np.zeros(len(rows), np.float32), where=days[rows, last] > 0)
    aggregate[:, aggregate_index["mean_weekly_activity"]] = daily * 7.0
    aggregate[:, aggregate_index["recent_activity"]] = last_rate * 7.0
    aggregate[:, aggregate_index["recent_historical_activity_ratio"]] = np.divide(last_rate, daily, out=np.zeros_like(last_rate), where=daily > 1e-6)
    for row in rows:
        idxs = np.flatnonzero(valid[row])
        rates = np.divide(activity[row, idxs], days[row, idxs], out=np.zeros(len(idxs), np.float32), where=days[row, idxs] > 0)
        if len(idxs) >= 2:
            aggregate[row, aggregate_index["activity_trend"]] = np.polyfit(idxs.astype(np.float32), rates, 1, w=np.sqrt(days[row, idxs]))[0]
        aggregate[row, aggregate_index["cumulative_inactive_weeks"]] = float(exposure[row, idxs][activity[row, idxs] <= 0].sum())
        streak = 0.0
        for pos in idxs[::-1]:
            if activity[row, pos] <= 0:
                streak += float(exposure[row, pos])
            else:
                break
        aggregate[row, aggregate_index["current_inactivity_streak"]] = streak
    metadata = dict(view.metadata)
    metadata["data_variant"] = "D3_both_safe"
    result = UnifiedHybridData(
        static=view.static.copy(),
        temporal=temporal,
        temporal_mask=view.temporal_mask.copy(),
        lengths=view.lengths.copy(),
        aggregate=aggregate,
        aggregate_available=view.aggregate_available.copy(),
        progress=view.progress.copy(),
        target=view.target.copy(),
        record_id=view.record_id.copy(),
        group_id=view.group_id.copy(),
        metadata=metadata,
    )
    result.validate()
    return result


def _ensure_phase8_source() -> None:
    """Expose Phase8 modules without writing into the kltn working tree."""
    import io
    import subprocess
    import tarfile

    import src

    from .protocol import AUTHORITY_REF

    bootstrap_kltn_namespace()
    extracted = ART / "authority_src"
    marker = extracted / "src" / "hybrid" / "phase8" / "data_variants.py"
    if not marker.exists():
        extracted.mkdir(parents=True, exist_ok=True)
        archive = subprocess.check_output(["git", "-C", str(KLTN), "archive", AUTHORITY_REF, "src/hybrid/phase8"])
        with tarfile.open(fileobj=io.BytesIO(archive), mode="r:") as bundle:
            bundle.extractall(extracted)
    extracted_src = str(extracted / "src")
    if extracted_src not in list(src.__path__):
        src.__path__.append(extracted_src)


def _pad_views(views: dict) -> dict:
    max_t = max(view.temporal.shape[1] for view in views.values())
    for view in views.values():
        pad = max_t - view.temporal.shape[1]
        if pad <= 0:
            continue
        object.__setattr__(view, "temporal", np.pad(view.temporal, ((0, 0), (0, pad), (0, 0))))
        object.__setattr__(view, "temporal_mask", np.pad(view.temporal_mask, ((0, 0), (0, pad))))
    return views


def load_raw_domain(domain: str):
    _ensure_phase8_source()
    if domain == "uci":
        from src.hybrid.data.uci import UCI_CATEGORICAL_CONTEXT, UCI_NUMERIC_CONTEXT, build_uci_combined
        from src.hybrid.phase7.data import build_uci_phase7_view

        frame, _ = build_uci_combined(KLTN / "data" / "raw" / "student-mat.csv", KLTN / "data" / "raw" / "student-por.csv")
        views = {stage: build_uci_phase7_view(frame, stage) for stage in UCI_STAGES}
        context = frame.rename(columns={"global_student_group": "group_id"})[
            ["record_id", "group_id", "target", *UCI_NUMERIC_CONTEXT, *UCI_CATEGORICAL_CONTEXT]
        ]
        return _pad_views(views), context, list(UCI_NUMERIC_CONTEXT), list(UCI_CATEGORICAL_CONTEXT)

    from src.hybrid.data.oulad import (
        OULAD_CATEGORICAL_CONTEXT,
        OULAD_NUMERIC_CONTEXT,
        build_compact_vle_daily,
        load_oulad_static_tables,
    )
    from src.hybrid.phase7.data import build_oulad_phase7_view

    _, _, base = load_oulad_static_tables(KLTN / "data" / "raw")
    daily = build_compact_vle_daily(KLTN / "data" / "raw", KLTN / "artifacts" / "hybrid" / "phase1" / "runtime")
    views = {}
    frames = []
    for stage in OULAD_PRIMARY:
        fraction = int(stage[:-3]) / 100.0
        eligible, view, _ = build_oulad_phase7_view(base, daily, fraction, str(KLTN / "data" / "raw"))
        views[stage] = _apply_d3(view)
        frames.append(eligible)
    context = (
        pd.concat(frames, ignore_index=True)
        .drop_duplicates("record_id")[["record_id", "group_id", "target", *OULAD_NUMERIC_CONTEXT, *OULAD_CATEGORICAL_CONTEXT]]
    )
    return _pad_views(views), context, list(OULAD_NUMERIC_CONTEXT), list(OULAD_CATEGORICAL_CONTEXT)


def cache_domain(domain: str):
    CACHE.mkdir(parents=True, exist_ok=True)
    marker = CACHE / f"{domain}_ready.json"
    if marker.exists():
        return
    views, context, numeric, categorical = load_raw_domain(domain)
    out = CACHE / domain
    out.mkdir(parents=True, exist_ok=True)
    for stage, view in views.items():
        stage_dir = out / stage
        stage_dir.mkdir(exist_ok=True)
        np.savez_compressed(
            stage_dir / "view.npz",
            temporal=view.temporal,
            temporal_mask=view.temporal_mask,
            lengths=view.lengths,
            aggregate=view.aggregate,
            aggregate_available=view.aggregate_available,
            progress=view.progress,
            target=view.target,
            record_id=view.record_id.astype(str),
            group_id=view.group_id.astype(str),
        )
        (stage_dir / "metadata.json").write_text(
            __import__("json").dumps(view.metadata, default=str), encoding="utf-8"
        )
    context.to_parquet(out / "context.parquet", index=False)
    write_json(marker, {"domain": domain, "stages": list(views), "numeric": numeric, "categorical": categorical})


def _load_cached_view(stage_dir: Path):
    from src.hybrid.phase7.contracts import UnifiedHybridData
    import json

    data = np.load(stage_dir / "view.npz", allow_pickle=False)
    metadata = json.loads((stage_dir / "metadata.json").read_text(encoding="utf-8"))
    n = len(data["record_id"])
    view = UnifiedHybridData(
        static=np.zeros((n, 0), np.float32),
        temporal=data["temporal"],
        temporal_mask=data["temporal_mask"],
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


def load_domain(domain: str):
    bootstrap_kltn_namespace()
    cache_domain(domain)
    marker = __import__("json").loads((CACHE / f"{domain}_ready.json").read_text(encoding="utf-8"))
    views = {stage: _load_cached_view(CACHE / domain / stage) for stage in marker["stages"]}
    context = pd.read_parquet(CACHE / domain / "context.parquet")
    return views, context, marker["numeric"], marker["categorical"]


def ensure_oulad_100pct() -> None:
    """Build cutoff-consistent 100pct view without rebuilding 20/35/50/75."""
    cache_domain("oulad")
    stage_dir = CACHE / "oulad" / "100pct"
    if (stage_dir / "view.npz").exists() and (stage_dir / "metadata.json").exists():
        return
    _ensure_phase8_source()
    from src.hybrid.data.oulad import (
        OULAD_CATEGORICAL_CONTEXT,
        OULAD_NUMERIC_CONTEXT,
        build_compact_vle_daily,
        load_oulad_static_tables,
    )
    from src.hybrid.phase7.data import build_oulad_phase7_view

    _, _, base = load_oulad_static_tables(KLTN / "data" / "raw")
    daily = build_compact_vle_daily(KLTN / "data" / "raw", KLTN / "artifacts" / "hybrid" / "phase1" / "runtime")
    eligible, view, _ = build_oulad_phase7_view(base, daily, 1.0, str(KLTN / "data" / "raw"))
    view = _apply_d3(view)
    stage_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        stage_dir / "view.npz",
        temporal=view.temporal,
        temporal_mask=view.temporal_mask,
        lengths=view.lengths,
        aggregate=view.aggregate,
        aggregate_available=view.aggregate_available,
        progress=view.progress,
        target=view.target,
        record_id=view.record_id.astype(str),
        group_id=view.group_id.astype(str),
    )
    extra = eligible.drop_duplicates("record_id")[["record_id", "group_id", "target", *OULAD_NUMERIC_CONTEXT, *OULAD_CATEGORICAL_CONTEXT]]
    extra.to_parquet(stage_dir / "eligible.parquet", index=False)
    import json as _json

    (stage_dir / "metadata.json").write_text(_json.dumps({**view.metadata, "stage": "100pct", "cutoff_fraction": 1.0}, default=str), encoding="utf-8")


def load_domain_phase4(domain: str):
    """Same FIT/STOP/VALID protocol; OULAD includes 100pct as an information state of the same model."""
    views, context, numeric, categorical = load_domain(domain)
    if domain != "oulad":
        return views, context, numeric, categorical
    ensure_oulad_100pct()
    views["100pct"] = _load_cached_view(CACHE / "oulad" / "100pct")
    extra_path = CACHE / "oulad" / "100pct" / "eligible.parquet"
    if extra_path.exists():
        extra = pd.read_parquet(extra_path)
        extra["record_id"] = extra.record_id.astype(str)
        context = context.copy()
        context["record_id"] = context.record_id.astype(str)
        context = pd.concat([context, extra], ignore_index=True).drop_duplicates("record_id")
    return _pad_views(views), context, numeric, categorical


def outer_holdout_ids(domain: str, outer_fold: int) -> tuple[list[str], list[str]]:
    """Return (train_ids, test_ids) for one outer fold. Test IDs must not be trained on."""
    from .protocol import split_paths

    frame = pd.read_parquet(split_paths()[f"{domain}_outer"])
    rec = frame.record_id.astype(str)
    test = rec[frame.outer_fold == outer_fold].tolist()
    train = rec[frame.outer_fold != outer_fold].tolist()
    if set(train) & set(test):
        raise RuntimeError("OUTER_TRAIN_TEST_OVERLAP")
    return train, test


def inner_partitions(domain: str, context: pd.DataFrame, inner_fold: int) -> tuple[list[str], list[str], list[str]]:
    from sklearn.model_selection import StratifiedGroupKFold

    inner = pd.read_parquet(split_paths()[f"{domain}_inner"])
    inner = inner[inner.outer_fold == DEVELOPMENT_OUTER_FOLD].copy()
    inner.record_id = inner.record_id.astype(str)
    valid_ids = set(inner.loc[inner.inner_fold == inner_fold, "record_id"])
    train_ids = set(inner.loc[inner.inner_fold != inner_fold, "record_id"])
    available = set(context.record_id.astype(str))
    valid_ids &= available
    train_ids &= available
    assert_disjoint(train_ids, valid_ids)
    assert_no_outer(train_ids | valid_ids, domain)
    frame = context[context.record_id.astype(str).isin(train_ids)].drop_duplicates("record_id").reset_index(drop=True)
    splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    y = frame.target.to_numpy()
    groups = frame.group_id.astype(str).to_numpy()
    fit_ids = stop_ids = None
    for fit, stop in splitter.split(frame, y, groups):
        if len(np.unique(y[fit])) == 2 and len(np.unique(y[stop])) == 2 and not (set(groups[fit]) & set(groups[stop])):
            fit_ids = frame.iloc[fit].record_id.astype(str).tolist()
            stop_ids = frame.iloc[stop].record_id.astype(str).tolist()
            break
    if fit_ids is None:
        raise RuntimeError("NO_FEASIBLE_FIT_STOP")
    valid = sorted(valid_ids)
    assert_disjoint(fit_ids, stop_ids, valid)
    assert_no_outer(set(fit_ids) | set(stop_ids) | set(valid), domain)
    return fit_ids, stop_ids, valid


def _temporal_summaries(temporal: np.ndarray, mask: np.ndarray) -> np.ndarray:
    n, _, channels = temporal.shape
    out = np.zeros((n, channels * 3), dtype=np.float32)
    for i in range(n):
        valid = mask[i]
        if not valid.any():
            continue
        values = temporal[i, valid]
        out[i, 0:channels] = values[-1]
        out[i, channels : 2 * channels] = values.mean(0)
        out[i, 2 * channels :] = values.max(0)
    return out


def scale_views(views: dict, context: pd.DataFrame, numeric: list[str], categorical: list[str], fit_ids: list[str], domain: str):
    from src.hybrid.contracts import MaskedStandardScaler
    from src.hybrid.training.data import ContextPreprocessor

    views = copy.deepcopy(views)
    fit_frame = context[context.record_id.astype(str).isin(fit_ids)].drop_duplicates("record_id")
    prep = ContextPreprocessor(numeric, categorical).fit(fit_frame)
    raw = prep.transform(context)
    static_map = {str(record_id): raw[i] for i, record_id in enumerate(context.record_id.astype(str))}
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
        view.aggregate[:] = (view.aggregate - mean) / std
    if domain == "oulad":
        xs, ms = [], []
        for view in views.values():
            lookup = {str(r): i for i, r in enumerate(view.record_id)}
            idx = [lookup[r] for r in fit_ids if r in lookup]
            if idx:
                xs.append(view.temporal[idx])
                ms.append(view.temporal_mask[idx])
        scaler = MaskedStandardScaler().fit(np.concatenate(xs), np.concatenate(ms))
        for view in views.values():
            view.temporal[:] = scaler.transform(view.temporal, view.temporal_mask)
    summaries: dict[str, dict[str, np.ndarray]] = {}
    for stage, view in views.items():
        values = _temporal_summaries(view.temporal, view.temporal_mask)
        lookup = {str(r): i for i, r in enumerate(view.record_id)}
        fit_idx = [lookup[r] for r in fit_ids if r in lookup]
        if fit_idx:
            mu = values[fit_idx].mean(0)
            sd = values[fit_idx].std(0)
            sd = np.where(sd < 1e-6, 1.0, sd)
        else:
            mu = np.zeros(values.shape[1], np.float32)
            sd = np.ones(values.shape[1], np.float32)
        summaries[stage] = {
            str(record_id): ((values[i] - mu) / sd).astype(np.float32)
            for i, record_id in enumerate(view.record_id.astype(str))
        }
    first = next(iter(views.values()))
    summary_dim = first.temporal.shape[2] * 3
    temporal_names = list(first.metadata.get("temporal_channels") or first.metadata.get("channels") or [f"ch{i}" for i in range(first.temporal.shape[2])])
    aggregate_names = list(first.metadata.get("aggregate_channels") or [f"agg{i}" for i in range(first.aggregate.shape[1])])
    contract = {
        "domain": domain,
        "static_numeric": numeric,
        "static_categorical": categorical,
        "aggregate": aggregate_names,
        "temporal_channels": temporal_names,
        "safe_summaries": [f"{name}__{stat}" for name in temporal_names for stat in ("last", "mean", "max")],
        "progress": True,
        "forbidden": list(FORBIDDEN_UCI if domain == "uci" else FORBIDDEN_OULAD),
        "cutoff_rule": "UCI stage mask; OULAD event_time < cutoff and observation_start <= event",
        "d3_applied": domain == "oulad",
        "fit_only_scaling": True,
        "baseline_receives_same_columns": True,
        "hybrid_privileged_features": [],
    }
    contract["hash"] = sha256_text(__import__("json").dumps(contract, sort_keys=True))
    return PreparedDomain(
        domain=domain,
        stages=tuple(views),
        views=views,
        context=context,
        numeric=numeric,
        categorical=categorical,
        static_map=static_map,
        summary_map=summaries,
        static_dim=int(prep.output_dim),
        temporal_dim=int(first.temporal.shape[2]),
        aggregate_dim=int(first.aggregate.shape[1]),
        summary_dim=int(summary_dim),
        feature_contract=contract,
    )


def baseline_frame(prepared: PreparedDomain, stage: str) -> pd.DataFrame:
    from src.hybrid.phase7.data import build_phase7_baseline_frame

    view = prepared.views[stage]
    aligned = prepared.context.assign(record_id=prepared.context.record_id.astype(str))
    aligned = aligned.drop_duplicates("record_id").set_index("record_id").loc[view.record_id.astype(str)].reset_index()
    frame = build_phase7_baseline_frame(aligned, view)
    predictors = [col for col in frame.columns if col not in {"record_id", "group_id", "target"}]
    leaked = [col for col in predictors if col.lower() in set(FORBIDDEN_UCI + FORBIDDEN_OULAD) or col in {"G1", "G2", "G3"}]
    if leaked:
        raise RuntimeError(f"BASELINE_FORBIDDEN_COLUMN:{leaked}")
    return frame


def feature_groups(frame: pd.DataFrame) -> dict[str, list[str]]:
    cols = [c for c in frame.columns if c not in {"record_id", "group_id", "target"}]
    static = [c for c in cols if not c.startswith("aggregate__") and not c.startswith("temporal__") and c != "progress"]
    aggregate = [c for c in cols if c.startswith("aggregate__")]
    summaries = [c for c in cols if c.startswith("temporal__") or c == "progress"]
    return {
        "static": static,
        "static_aggregate": static + aggregate,
        "full": static + aggregate + summaries,
    }


def permute_temporal(temporal: np.ndarray, mask: np.ndarray, mode: str, seed: int) -> np.ndarray:
    if mode == "identity":
        return temporal.copy()
    out = temporal.copy()
    rng = np.random.default_rng(seed)
    for i, valid in enumerate(mask):
        idx = np.flatnonzero(valid)
        if len(idx) < 2:
            continue
        if mode == "reverse":
            out[i, idx] = temporal[i, idx[::-1]]
        elif mode == "shuffle":
            shuffled = idx.copy()
            rng.shuffle(shuffled)
            out[i, idx] = temporal[i, shuffled]
        else:
            raise ValueError(mode)
    out[~mask] = 0.0
    return out


def final100_length_diagnostic() -> dict[str, Any]:
    bootstrap_kltn_namespace()
    from src.hybrid.data.oulad import load_oulad_static_tables
    from sklearn.metrics import average_precision_score, roc_auc_score

    _, _, base = load_oulad_static_tables(KLTN / "data" / "raw")
    course_end = base.module_presentation_length.to_numpy(np.int64)
    unregistered = base.date_unregistration.where(base.date_unregistration.notna(), pd.Series(course_end, index=base.index)).to_numpy(np.float64)
    unregistered = np.where(np.isfinite(unregistered), unregistered, course_end)
    length = np.clip(np.minimum(course_end, unregistered).astype(np.int64), 0, course_end)
    weeks = np.maximum(0, np.ceil(length / 7.0)).astype(np.int64)
    withdrawn = (base.final_result.astype(str) == "Withdrawn").to_numpy()
    risk = base.target.to_numpy()
    fail = (base.final_result.astype(str) == "Fail").to_numpy()
    completed = base.final_result.astype(str).isin(["Pass", "Distinction", "Fail"]).to_numpy()
    short = weeks <= 20
    score = -weeks.astype(np.float64)
    return {
        "n": int(len(base)),
        "withdrawn_rate": float(withdrawn.mean()),
        "risk_rate": float(risk.mean()),
        "mean_weeks_by_label": {
            str(label): float(weeks[base.final_result.astype(str) == label].mean())
            for label in sorted(base.final_result.astype(str).unique())
        },
        "short_history_withdrawn_rate": float(withdrawn[short].mean()) if short.any() else None,
        "length_pr_auc_withdrawn": float(average_precision_score(withdrawn, score)),
        "length_roc_auc_withdrawn": float(roc_auc_score(withdrawn, score)),
        "length_pr_auc_risk": float(average_precision_score(risk, score)),
        "fail_vs_pass_length_pr_auc": float(average_precision_score(fail[completed], score[completed])) if completed.any() else None,
        "flagged_shortcut_risk": bool(short.any() and withdrawn[short].mean() >= 0.95),
        "used_for_architecture_selection": False,
        "outer_test_used": False,
    }
