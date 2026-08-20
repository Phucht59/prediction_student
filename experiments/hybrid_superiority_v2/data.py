"""Raw-to-stage pipeline, group-safe splits, FIT-only scaling. No kltn dependency."""
from __future__ import annotations

import copy
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

from src.prediction.contracts import assert_binary_target, uci_risk_target
from src.prediction.data.common import UnifiedHybridData
from src.prediction.data.oulad import load_oulad_static_tables
from src.prediction.data.oulad_features import (
    OULAD_CATEGORICAL_CONTEXT,
    OULAD_NUMERIC_CONTEXT,
    STATE_FRACTIONS,
    apply_d3_variant,
)
from src.prediction.data.preprocessing import ContextPreprocessor, MaskedStandardScaler
from src.prediction.data.uci import (
    UCI_CATEGORICAL_CONTEXT,
    UCI_FORBIDDEN_PREDICTORS,
    UCI_NUMERIC_CONTEXT,
    UCI_QUASI_IDENTITY_FIELDS,
    build_uci_combined,
)

from .io_utils import sha256_file, sha256_json, sha256_text, write_json
from .oulad_build import augment_temporal_deltas, build_oulad_cutoff_view, build_vle_daily
from .paths import CACHE_DIR, DATA_ROOT, MANIFEST_DIR, ensure_dirs
from .protocol import (
    DEVELOPMENT_OUTER_FOLD,
    FORBIDDEN_OULAD,
    FORBIDDEN_UCI,
    N_INNER,
    N_OUTER,
    OULAD_STAGES,
    RAW_SHA256,
    SPLIT_SEED,
    UCI_STAGES,
    protocol_hash,
    stages_for,
)


UCI_PROGRESS = {"S0": 0.0, "S1": 0.5, "S2": 1.0}


def verify_raw_checksums() -> dict[str, str]:
    observed = {}
    for name, expected in RAW_SHA256.items():
        path = DATA_ROOT / name
        if not path.exists():
            raise FileNotFoundError(f"RAW_MISSING:{path}. Set DATA_ROOT or place official files under data/raw.")
        digest = sha256_file(path)
        if digest != expected:
            raise RuntimeError(f"RAW_HASH_MISMATCH:{name}:{digest}:{expected}")
        observed[name] = digest
    return observed


def _stable_id(*parts: Any, length: int = 24) -> str:
    import hashlib

    return hashlib.sha256("|".join(str(x).strip() for x in parts).encode()).hexdigest()[:length]


def build_uci_views(uci_df: pd.DataFrame) -> dict[str, UnifiedHybridData]:
    """G1/G2 live only in the temporal branch. Aggregate is disabled for UCI Hybrid."""
    n = len(uci_df)
    g1 = uci_df.G1.to_numpy(np.float32) / 20.0
    g2 = uci_df.G2.to_numpy(np.float32) / 20.0
    views = {}
    for stage in UCI_STAGES:
        temporal = np.zeros((n, 2, 2), dtype=np.float32)
        mask = np.zeros((n, 2), dtype=bool)
        if stage in {"S1", "S2"}:
            temporal[:, 0, 0] = g1
            mask[:, 0] = True
        if stage == "S2":
            temporal[:, 1, 0] = g2
            temporal[:, 1, 1] = g2 - g1
            mask[:, 1] = True
        temporal[~mask] = 0.0
        view = UnifiedHybridData(
            static=np.zeros((n, 0), np.float32),
            temporal=temporal,
            temporal_mask=mask,
            lengths=mask.sum(1).astype(np.int64),
            aggregate=np.zeros((n, 1), np.float32),
            aggregate_available=np.zeros(n, np.int8),
            progress=np.full(n, UCI_PROGRESS[stage], np.float32),
            target=uci_df.target.to_numpy(np.int64),
            record_id=uci_df.record_id.to_numpy(str),
            group_id=uci_df.global_student_group.to_numpy(str),
            metadata={
                "dataset": "uci_combined",
                "stage": stage,
                "temporal_channels": ["grade_norm", "grade_delta"],
                "aggregate_channels": ["disabled"],
                "g1_g2_in_aggregate": False,
                "g1_g2_in_static": False,
                "g3_predictor": False,
            },
        )
        view.validate()
        if stage == "S0":
            assert not mask.any()
        if stage == "S1":
            assert mask[:, 0].all() and not mask[:, 1].any()
        if stage == "S2":
            assert mask.all()
        views[stage] = view
    return views


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


def prepare_uci() -> dict[str, Any]:
    ensure_dirs()
    frame, summary = build_uci_combined(DATA_ROOT / "student-mat.csv", DATA_ROOT / "student-por.csv")
    leaked = [c for c in UCI_NUMERIC_CONTEXT + UCI_CATEGORICAL_CONTEXT if c in UCI_FORBIDDEN_PREDICTORS]
    if leaked:
        raise RuntimeError(f"UCI_CONTEXT_LEAK:{leaked}")
    views = _pad_views(build_uci_views(frame))
    cols = list(dict.fromkeys(["record_id", "group_id", "target", "G1", "G2", *UCI_NUMERIC_CONTEXT, *UCI_CATEGORICAL_CONTEXT]))
    context = frame.rename(columns={"global_student_group": "group_id"})[cols].copy()
    out = CACHE_DIR / "uci"
    out.mkdir(parents=True, exist_ok=True)
    _save_views(out, views)
    context.to_parquet(out / "context.parquet", index=False)
    manifest = {
        "dataset": "uci",
        **summary,
        "prevalence": float(frame.target.mean()),
        "n_risk": int(frame.target.sum()),
        "stages": {stage: {"n": int(len(views[stage].record_id)), "prevalence": float(views[stage].target.mean())} for stage in views},
        "g1_g2_hybrid_branch": "temporal_only",
        "absences_used": False,
        "protocol_hash": protocol_hash(),
    }
    write_json(out / "manifest.json", manifest)
    return manifest


def prepare_oulad() -> dict[str, Any]:
    ensure_dirs()
    _, _, base = load_oulad_static_tables(DATA_ROOT)
    print("oulad static enrollments", len(base), "students", int(base.group_id.nunique()), flush=True)
    print("building vle daily cache", flush=True)
    daily = build_vle_daily(DATA_ROOT)
    print("vle daily rows", len(daily), flush=True)
    views = {}
    frames = []
    audits = {}
    for stage in OULAD_STAGES:
        print("building stage", stage, flush=True)
        eligible, view, audit = build_oulad_cutoff_view(base, daily, STATE_FRACTIONS[stage], DATA_ROOT)
        view = apply_d3_variant(view)
        view = augment_temporal_deltas(view)
        views[stage] = view
        frames.append(eligible)
        audits[stage] = {
            **audit,
            "n": int(len(eligible)),
            "n_groups": int(eligible.group_id.nunique()),
            "prevalence": float(eligible.target.mean()),
            "final_result_counts": eligible.final_result.astype(str).value_counts().to_dict() if "final_result" in eligible.columns else {},
        }
    views = _pad_views(views)
    context_cols = ["record_id", "group_id", "target", "code_module", "code_presentation", *OULAD_NUMERIC_CONTEXT, *OULAD_CATEGORICAL_CONTEXT]
    extra = [c for c in ["id_student", "final_result"] if c in frames[0].columns]
    context = (
        pd.concat(frames, ignore_index=True)
        .drop_duplicates("record_id")[list(dict.fromkeys(context_cols + extra))]
    )
    leaked = [c for c in context.columns if c in FORBIDDEN_OULAD and c not in {"target", "final_result"}]
    # final_result is kept for sensitivity audits only; never passed as a model feature.
    out = CACHE_DIR / "oulad"
    out.mkdir(parents=True, exist_ok=True)
    _save_views(out, views)
    context.to_parquet(out / "context.parquet", index=False)
    manifest = {
        "dataset": "oulad",
        "n_enrollments": int(len(base)),
        "n_students": int(base.group_id.nunique()),
        "prevalence": float(base.target.mean()),
        "stages": audits,
        "cutoff_rule": "observation_start <= event_time < cutoff",
        "vle_daily_rows": int(len(daily)),
        "protocol_hash": protocol_hash(),
        "forbidden_kept_out_of_features": list(FORBIDDEN_OULAD),
        "label_column_in_context_for_sensitivity_only": True,
    }
    write_json(out / "manifest.json", manifest)
    return manifest


def _save_views(out: Path, views: dict[str, UnifiedHybridData]) -> None:
    for stage, view in views.items():
        stage_dir = out / stage
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
        (stage_dir / "metadata.json").write_text(json.dumps(view.metadata, default=str), encoding="utf-8")


def _load_view(stage_dir: Path) -> UnifiedHybridData:
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


def load_cached(domain: str) -> tuple[dict[str, UnifiedHybridData], pd.DataFrame, dict]:
    root = CACHE_DIR / domain
    marker = root / "manifest.json"
    if not marker.exists():
        raise FileNotFoundError(f"cache missing for {domain}; run prepare")
    manifest = json.loads(marker.read_text(encoding="utf-8"))
    views = {stage: _load_view(root / stage) for stage in stages_for(domain)}
    context = pd.read_parquet(root / "context.parquet")
    context["record_id"] = context.record_id.astype(str)
    context["group_id"] = context.group_id.astype(str)
    return views, context, manifest


def make_splits(domain: str) -> dict[str, Any]:
    """Nested StratifiedGroupKFold. Outer fold 0 test is the confirmation firewall."""
    _, context, _ = load_cached(domain)
    frame = context.drop_duplicates("record_id").reset_index(drop=True)
    y = frame.target.to_numpy()
    groups = frame.group_id.astype(str).to_numpy()
    record_ids = frame.record_id.astype(str).to_numpy()
    outer_rows = []
    inner_rows = []
    splitter = StratifiedGroupKFold(n_splits=N_OUTER, shuffle=True, random_state=SPLIT_SEED)
    for outer_fold, (train_idx, test_idx) in enumerate(splitter.split(frame, y, groups)):
        for i in test_idx:
            outer_rows.append({"record_id": record_ids[i], "group_id": groups[i], "target": int(y[i]), "outer_fold": outer_fold, "role": "test"})
        train_frame = frame.iloc[train_idx].reset_index(drop=True)
        inner = StratifiedGroupKFold(n_splits=N_INNER, shuffle=True, random_state=SPLIT_SEED)
        ty = train_frame.target.to_numpy()
        tg = train_frame.group_id.astype(str).to_numpy()
        tr = train_frame.record_id.astype(str).to_numpy()
        for inner_fold, (_, valid_idx) in enumerate(inner.split(train_frame, ty, tg)):
            valid_set = set(valid_idx)
            for j in range(len(train_frame)):
                inner_rows.append(
                    {
                        "record_id": tr[j],
                        "group_id": tg[j],
                        "target": int(ty[j]),
                        "outer_fold": outer_fold,
                        "inner_fold": inner_fold,
                        "role": "valid" if j in valid_set else "train",
                    }
                )
        test_groups = set(groups[test_idx])
        train_groups = set(tg)
        if test_groups & train_groups:
            raise RuntimeError("OUTER_GROUP_OVERLAP")
    outer_df = pd.DataFrame(outer_rows)
    inner_df = pd.DataFrame(inner_rows)
    split_dir = CACHE_DIR / "splits"
    split_dir.mkdir(parents=True, exist_ok=True)
    outer_path = split_dir / f"{domain}_outer.parquet"
    inner_path = split_dir / f"{domain}_inner.parquet"
    outer_df.to_parquet(outer_path, index=False)
    inner_df.to_parquet(inner_path, index=False)
    payload = {
        "domain": domain,
        "n_outer": N_OUTER,
        "n_inner": N_INNER,
        "split_seed": SPLIT_SEED,
        "n_records": int(len(frame)),
        "n_groups": int(frame.group_id.nunique()),
        "outer_sha256": sha256_file(outer_path),
        "inner_sha256": sha256_file(inner_path),
        "development_outer_fold": DEVELOPMENT_OUTER_FOLD,
        "outer_test_excluded_from_development": True,
    }
    write_json(split_dir / f"{domain}_split_lock.json", payload)
    return payload


def outer_test_ids(domain: str, outer_fold: int = DEVELOPMENT_OUTER_FOLD) -> set[str]:
    frame = pd.read_parquet(CACHE_DIR / "splits" / f"{domain}_outer.parquet")
    return set(frame.loc[frame.outer_fold == outer_fold, "record_id"].astype(str))


def assert_no_outer(ids, domain: str, outer_fold: int = DEVELOPMENT_OUTER_FOLD) -> None:
    blocked = set(map(str, ids)) & outer_test_ids(domain, outer_fold)
    if blocked:
        raise RuntimeError(f"OUTER_FIREWALL_VIOLATION:{domain}:{len(blocked)}")


def inner_partitions(domain: str, inner_fold: int, outer_fold: int = DEVELOPMENT_OUTER_FOLD) -> tuple[list[str], list[str], list[str]]:
    inner = pd.read_parquet(CACHE_DIR / "splits" / f"{domain}_inner.parquet")
    inner = inner[inner.outer_fold == outer_fold].drop_duplicates(["record_id", "inner_fold"])
    valid_ids = set(inner.loc[(inner.inner_fold == inner_fold) & (inner.role == "valid"), "record_id"].astype(str))
    train_ids = set(inner.loc[(inner.inner_fold == inner_fold) & (inner.role == "train"), "record_id"].astype(str))
    if not valid_ids or not train_ids:
        raise RuntimeError("EMPTY_INNER_PARTITION")
    assert_no_outer(train_ids | valid_ids, domain, outer_fold)
    if train_ids & valid_ids:
        raise RuntimeError("INNER_TRAIN_VALID_OVERLAP")
    context = pd.read_parquet(CACHE_DIR / domain / "context.parquet")
    rest = context[context.record_id.astype(str).isin(train_ids)].drop_duplicates("record_id").reset_index(drop=True)
    splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    y = rest.target.to_numpy()
    groups = rest.group_id.astype(str).to_numpy()
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
    assert_no_outer(set(fit_ids) | set(stop_ids) | set(valid), domain, outer_fold)
    return fit_ids, stop_ids, valid


@dataclass
class PreparedDomain:
    domain: str
    stages: tuple[str, ...]
    views: dict[str, UnifiedHybridData]
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
    fit_ids: tuple[str, ...] = field(default_factory=tuple)


def _temporal_summaries(temporal: np.ndarray, mask: np.ndarray) -> np.ndarray:
    n, t, c = temporal.shape
    last = np.zeros((n, c), np.float32)
    mean = np.zeros((n, c), np.float32)
    mx = np.zeros((n, c), np.float32)
    std = np.zeros((n, c), np.float32)
    slope = np.zeros((n, c), np.float32)
    count = mask.sum(1).astype(np.float32)
    time = np.arange(t, dtype=np.float32)
    for i in range(n):
        valid = mask[i]
        if not valid.any():
            continue
        values = temporal[i, valid]
        last[i] = values[-1]
        mean[i] = values.mean(0)
        mx[i] = values.max(0)
        std[i] = values.std(0)
        if valid.sum() >= 2:
            tt = time[valid]
            tt = tt - tt.mean()
            denom = float((tt ** 2).sum())
            if denom > 0:
                slope[i] = (tt[:, None] * (values - values.mean(0))).sum(0) / denom
    return np.concatenate([last, mean, mx, std, slope, count[:, None]], axis=1).astype(np.float32)


def scale_views(domain: str, fit_ids: list[str]) -> PreparedDomain:
    views, context, _manifest = load_cached(domain)
    views = copy.deepcopy(views)
    if domain == "uci":
        numeric, categorical = list(UCI_NUMERIC_CONTEXT), list(UCI_CATEGORICAL_CONTEXT)
    else:
        numeric, categorical = list(OULAD_NUMERIC_CONTEXT), list(OULAD_CATEGORICAL_CONTEXT)
    fit_frame = context[context.record_id.astype(str).isin(fit_ids)].drop_duplicates("record_id")
    prep = ContextPreprocessor(numeric, categorical).fit(fit_frame)
    raw = prep.transform(context.drop_duplicates("record_id"))
    ids = context.drop_duplicates("record_id").record_id.astype(str).to_numpy()
    static_map = {str(record_id): raw[i] for i, record_id in enumerate(ids)}
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
        view.aggregate[~view.aggregate_available.astype(bool)] = 0
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
            mu, sd = values[fit_idx].mean(0), values[fit_idx].std(0)
            sd = np.where(sd < 1e-6, 1.0, sd)
        else:
            mu = np.zeros(values.shape[1], np.float32)
            sd = np.ones(values.shape[1], np.float32)
        summaries[stage] = {
            str(record_id): ((values[i] - mu) / sd).astype(np.float32)
            for i, record_id in enumerate(view.record_id.astype(str))
        }
    first = next(iter(views.values()))
    contract = {
        "domain": domain,
        "static_numeric": numeric,
        "static_categorical": categorical,
        "g1_g2_in_hybrid_static": False,
        "g1_g2_in_hybrid_aggregate": False,
        "fit_only_scaling": True,
        "forbidden": list(FORBIDDEN_UCI if domain == "uci" else FORBIDDEN_OULAD),
        "protocol_hash": protocol_hash(),
    }
    contract["hash"] = sha256_json(contract)
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
        summary_dim=int(next(iter(summaries[next(iter(summaries))].values())).shape[0]),
        feature_contract=contract,
        fit_ids=tuple(fit_ids),
    )


def hybrid_forbidden_columns(columns) -> list[str]:
    lowered = {c.lower() for c in columns}
    names = []
    seen = set()
    for name in list(FORBIDDEN_UCI) + list(FORBIDDEN_OULAD):
        key = name.lower()
        if key in lowered and key not in seen:
            names.append(name)
            seen.add(key)
    return names


def baseline_frame(prepared: PreparedDomain, stage: str) -> pd.DataFrame:
    """Panel A: same raw information, representation-native for trees/linear models."""
    view = prepared.views[stage]
    ctx = prepared.context.drop_duplicates("record_id").copy()
    ctx["record_id"] = ctx.record_id.astype(str)
    aligned = ctx.set_index("record_id").loc[view.record_id.astype(str)].reset_index()
    frame = aligned[["record_id", "group_id", "target", *prepared.numeric, *prepared.categorical]].copy()
    if prepared.domain == "uci":
        if stage in {"S1", "S2"}:
            frame["grade_g1"] = aligned["G1"].astype(np.float32)
        if stage == "S2":
            frame["grade_g2"] = aligned["G2"].astype(np.float32)
        # Hybrid does not receive these as tabular; trees/LR do. Same raw information.
    else:
        summaries = np.stack([prepared.summary_map[stage][str(r)] for r in view.record_id.astype(str)])
        for j in range(view.aggregate.shape[1]):
            frame[f"aggregate__{j}"] = view.aggregate[:, j]
        for j in range(summaries.shape[1]):
            frame[f"temporal_summary__{j}"] = summaries[:, j]
        if "code_module" in aligned.columns:
            frame["code_module"] = aligned["code_module"]
        if "code_presentation" in aligned.columns:
            frame["code_presentation"] = aligned["code_presentation"]
    frame["progress"] = view.progress
    predictors = [c for c in frame.columns if c not in {"record_id", "group_id", "target", "final_result", "id_student", "G1", "G2"}]
    leaked = [c for c in predictors if c.lower() in {x.lower() for x in FORBIDDEN_OULAD + ("G3", "absences")}]
    if leaked:
        raise RuntimeError(f"BASELINE_FORBIDDEN:{leaked}")
    if prepared.domain == "uci" and stage == "S0" and any(c.startswith("grade_g") for c in frame.columns):
        raise RuntimeError("S0_HAS_GRADES")
    return frame


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


def prepare_all() -> dict[str, Any]:
    ensure_dirs()
    raw = verify_raw_checksums()
    uci = prepare_uci()
    oulad = prepare_oulad()
    splits = {domain: make_splits(domain) for domain in ("uci", "oulad")}
    payload = {"raw": raw, "uci": uci, "oulad": oulad, "splits": splits, "protocol_hash": protocol_hash()}
    write_json(MANIFEST_DIR / "data_lock.json", payload)
    return payload
