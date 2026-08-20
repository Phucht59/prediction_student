"""Nested CV: official outer folds are TEST only. Threshold and early stop from STOP inside the train side. Frozen TRAINING_CONFIG. One Hybrid spec."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.cv5.run_cv5 import CACHE, HPARAMS, OUT, _stages
from experiments.hybrid_vnext.data import outer_holdout_ids
from experiments.hybrid_vnext.protocol import verify_split_hashes
from experiments.imbalance.data_build import build_oulad_stage, build_uci_stage, oulad_context, raw_dir, uci_context, _load_batch, _save_batch
from experiments.imbalance.evaluation import select_stop_threshold
from experiments.imbalance.samplers import subset_batch
from experiments.imbalance.train_hybrid import predict_scores, train_one
from experiments.validation.metrics import full_metrics
from experiments.validation.scoring import fit_sklearn_baselines, pack_tabular
from sklearn.metrics import average_precision_score

NEST = OUT / "nested_outer"


def _fit_stop(context: pd.DataFrame, train_ids: list[str]) -> tuple[list[str], list[str]]:
    rest = context[context.record_id.astype(str).isin(set(train_ids))].drop_duplicates("record_id").reset_index(drop=True)
    splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    y = rest.target.to_numpy()
    g = rest.group_id.astype(str).to_numpy()
    for fit, stop in splitter.split(rest, y, g):
        if len(np.unique(y[fit])) == 2 and len(np.unique(y[stop])) == 2 and not (set(g[fit]) & set(g[stop])):
            return rest.iloc[fit].record_id.astype(str).tolist(), rest.iloc[stop].record_id.astype(str).tolist()
    raise RuntimeError("NO_FEASIBLE_FIT_STOP")


def load_outer_fold(dataset: str, outer_fold: int, *, vle=None) -> dict:
    verify_split_hashes()
    context = uci_context() if dataset == "uci" else oulad_context()
    train_ids, test_ids = outer_holdout_ids(dataset, outer_fold)
    train_ids = [i for i in train_ids if i in set(context.record_id.astype(str))]
    test_ids = [i for i in test_ids if i in set(context.record_id.astype(str))]
    if set(train_ids) & set(test_ids):
        raise RuntimeError("OUTER_TRAIN_TEST_OVERLAP")
    fit_ids, stop_ids = _fit_stop(context, train_ids)
    if set(fit_ids) & set(test_ids) or set(stop_ids) & set(test_ids):
        raise RuntimeError("STOP_OR_FIT_IN_OUTER")
    keep = list(dict.fromkeys([*fit_ids, *stop_ids, *test_ids]))
    preprocessor = None
    train_stages, stop_stages, test_stages = {}, {}, {}
    for stage in _stages(dataset):
        a, b, c = (CACHE / f"nested_{dataset}_o{outer_fold}_{stage}_{p}.npz" for p in ("train", "stop", "test"))
        if a.is_file() and b.is_file() and c.is_file():
            print(f"  cache nested {dataset} outer {outer_fold} {stage}", flush=True)
            train_stages[stage] = _load_batch(a)
            stop_stages[stage] = _load_batch(b)
            test_stages[stage] = _load_batch(c)
            continue
        print(f"  build nested {dataset} outer {outer_fold} {stage}", flush=True)
        if dataset == "uci":
            full = build_uci_stage(stage, fit_ids, keep)
        else:
            full, preprocessor = build_oulad_stage(stage, fit_ids, keep, preprocessor=preprocessor, vle_daily=vle)
        train_stages[stage] = subset_batch(full, fit_ids)
        stop_stages[stage] = subset_batch(full, stop_ids)
        test_stages[stage] = subset_batch(full, test_ids)
        _save_batch(a, train_stages[stage])
        _save_batch(b, stop_stages[stage])
        _save_batch(c, test_stages[stage])
    return {"train_stages": train_stages, "stop_stages": stop_stages, "test_stages": test_stages, "n_test": len(test_ids)}


def run_nested(dataset: str, outer_folds: range, seed: int = 42) -> pd.DataFrame:
    hparams = HPARAMS["uci" if dataset == "uci" else "oulad"]
    vle = None
    if dataset == "oulad":
        from src.prediction.data.oulad_features import build_vle_daily

        vle = build_vle_daily(raw_dir())
    rows = []
    for of in outer_folds:
        print("NESTED OUTER", dataset, of, flush=True)
        packed = load_outer_fold(dataset, of, vle=vle)
        result = train_one(packed["train_stages"], packed["stop_stages"], packed["test_stages"], hparams, seed=seed, keep_model=False)
        for stage in _stages(dataset):
            test = packed["test_stages"][stage]
            train = packed["train_stages"][stage]
            hp = result["valid_scores"][stage]
            t = result["stage_metrics"][stage]["threshold"]
            base = fit_sklearn_baselines(train, test, seed=seed, feature_mode="tabular")
            x_stop = pack_tabular(packed["stop_stages"][stage])
            y_stop = packed["stop_stages"][stage].target
            thresh = {"Hybrid": t}
            for name in ("LR", "RF"):
                thresh[name] = select_stop_threshold(y_stop, base["models"][name].predict_proba(x_stop)[:, 1])
            for name, scores in (("Hybrid", hp), ("LR", base["LR"]), ("RF", base["RF"])):
                m = full_metrics(test.target, scores, threshold=thresh[name])
                rows.append(
                    {
                        "dataset": dataset,
                        "outer_fold": of,
                        "information_level": stage,
                        "model": name,
                        "seed": seed,
                        "hpo_on_outer": False,
                        "threshold_from": "STOP",
                        **m,
                    }
                )
            print(
                f"  {stage} Hybrid {average_precision_score(test.target, hp):.4f} "
                f"RF {average_precision_score(test.target, base['RF']):.4f} "
                f"LR {average_precision_score(test.target, base['LR']):.4f}",
                flush=True,
            )
    return pd.DataFrame(rows)


def case_studies() -> None:
    path = OUT / "scientific" / "scores.parquet"
    alt = ROOT / "artifacts" / "experiments" / "validation" / "scores_uci.parquet"
    src = path if path.is_file() else alt
    if not src.is_file():
        return
    scores = pd.read_parquet(src)
    h = scores[(scores.model == "Hybrid")].copy()
    h["pred"] = (h.score >= h.threshold).astype(int)
    h["error"] = "TN"
    h.loc[(h.target == 1) & (h.pred == 1), "error"] = "TP"
    h.loc[(h.target == 0) & (h.pred == 1), "error"] = "FP"
    h.loc[(h.target == 1) & (h.pred == 0), "error"] = "FN"
    rows = []
    for (dataset, stage), part in h.groupby(["dataset", "information_level"]):
        for kind in ("TP", "FP", "FN", "TN"):
            chunk = part[part.error == kind]
            if chunk.empty:
                continue
            pick = chunk.sort_values("score", ascending=(kind in {"FN", "TN"})).head(3)
            for _, rec in pick.iterrows():
                rows.append(
                    {
                        "dataset": dataset,
                        "information_level": stage,
                        "error": kind,
                        "record_id": rec.record_id,
                        "target": int(rec.target),
                        "score": float(rec.score),
                        "threshold": float(rec.threshold),
                    }
                )
    if rows:
        NEST.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows).to_csv(NEST / "error_case_studies.csv", index=False)


def permutation_uci() -> None:
    from experiments.cv5.run_cv5 import load_cv_fold
    from experiments.imbalance.samplers import PackedBatch
    from experiments.imbalance.train_hybrid import predict_scores

    packed = load_cv_fold("uci", 0)
    result = train_one(packed["train_stages"], packed["stop_stages"], packed["valid_stages"], HPARAMS["uci"], seed=42, keep_model=True)
    model = result["model"]
    import torch

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    rng = np.random.default_rng(0)
    rows = []
    for stage, valid in packed["valid_stages"].items():
        base = float(average_precision_score(valid.target, predict_scores(model, valid, batch_size=32)))

        def shuffled(batch, field):
            payload = dict(batch.__dict__)
            arr = np.array(payload[field], copy=True)
            payload[field] = arr[rng.permutation(len(arr))]
            return PackedBatch(**payload)

        for field in ("static", "aggregate", "temporal"):
            dropped = shuffled(valid, field)
            val = float(average_precision_score(valid.target, predict_scores(model, dropped, batch_size=32)))
            rows.append({"stage": stage, "block": field, "base_pr_auc": base, "shuffled_pr_auc": val, "delta": base - val})
    pd.DataFrame(rows).to_csv(NEST / "permutation_uci_fold0.csv", index=False)
    print("permutation wrote", flush=True)


def main() -> int:
    NEST.mkdir(parents=True, exist_ok=True)
    uci = run_nested("uci", range(5), seed=42)
    uci.to_csv(NEST / "uci_nested_outer.csv", index=False)
    try:
        oulad = run_nested("oulad", range(3), seed=42)
        oulad.to_csv(NEST / "oulad_nested_outer.csv", index=False)
        all_rows = pd.concat([uci, oulad], ignore_index=True)
    except Exception as exc:
        print("OULAD nested outer failed:", exc, flush=True)
        all_rows = uci
    all_rows.to_csv(NEST / "nested_outer_metrics.csv", index=False)
    case_studies()
    try:
        permutation_uci()
    except Exception as exc:
        print("permutation failed", exc, flush=True)
    print("WROTE", NEST)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
