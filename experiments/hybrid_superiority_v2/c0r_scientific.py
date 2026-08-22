"""C0-R scientific completion: same integrity bar as C4-STAR, C0-R topology only.

Time/quality: 16 OULAD + 8 UCI extra HPO trials, 20/24 epochs, batch 256, no KD, 3x3 robust.
GPU software cap 80C. Does not open outer test. Does not promote serving Hybrid.
"""
from __future__ import annotations

import json
import os
import time
import traceback

os.environ.setdefault("PYTHONUNBUFFERED", "1")
os.environ.setdefault("HS_V2_GPU_TREES", "0")
os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")

import numpy as np
import pandas as pd

from .data import inner_partitions, scale_views
from .gate import evaluate_development_gate
from .hpo import _make_study, _study_name, evaluate_hybrid, sample_hybrid
from .io_utils import git_commit, utc_now, write_json
from .metrics import selection_objective
from .model import make_config
from .paths import OOF_DIR, REPORT_ROOT, RUN_DIR, ensure_dirs
from .protocol import SCREEN_FOLD, SCREEN_SEED, SEEDS_ROBUST, protocol_hash
from .status import write_status
from .thermal import snapshot, wait_if_hot

CANDIDATE = "C0-R"
OULAD_HPO_TRIALS = 16
UCI_HPO_TRIALS = 8
OULAD_EPOCHS = 20
UCI_EPOCHS = 24
OULAD_BATCH = 256
STATE = RUN_DIR / "c0r_scientific_state.json"


def _log(msg: str) -> None:
    print(f"[{utc_now()}] {msg}", flush=True)


def _boost() -> None:
    try:
        import torch

        torch.backends.cudnn.benchmark = True
        torch.set_float32_matmul_precision("high")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            _log(f"CUDA {torch.cuda.get_device_name(0)}")
    except Exception as exc:
        _log(f"cuda {exc}")
    try:
        import psutil

        p = psutil.Process()
        p.nice(psutil.HIGH_PRIORITY_CLASS if hasattr(psutil, "HIGH_PRIORITY_CLASS") else psutil.ABOVE_NORMAL_PRIORITY_CLASS)
        _log("priority HIGH")
    except Exception:
        pass


def _load_state() -> dict:
    if STATE.exists():
        return json.loads(STATE.read_text(encoding="utf-8"))
    return {"completed": []}


def _save_state(st: dict) -> None:
    write_json(STATE, st)


def load_scientific_ceiling(domain: str) -> dict[str, float]:
    from .paths import MANIFEST_DIR

    if domain == "oulad":
        for path in (
            MANIFEST_DIR / "baseline_lock_oulad_v21.json",
            RUN_DIR / "baseline_lock_oulad_v21.json",
            RUN_DIR / "baseline_lock_oulad.json",
        ):
            if path.exists():
                lock = json.loads(path.read_text(encoding="utf-8"))
                return {k: float(v["ap"]) for k, v in lock["stage_best_ap"].items()}
    path = RUN_DIR / f"baseline_lock_{domain}.json"
    lock = json.loads(path.read_text(encoding="utf-8"))
    return {k: float(v["ap"]) for k, v in lock["stage_best_ap"].items()}


def _force_train_kw(train_kw: dict, domain: str) -> dict:
    train_kw = dict(train_kw)
    train_kw["lambda_kd"] = 0.0
    train_kw["use_ema"] = True
    train_kw["multiprefix"] = True
    train_kw["patience"] = 8
    if domain == "oulad":
        train_kw["max_epochs"] = OULAD_EPOCHS
        train_kw["batch_size"] = OULAD_BATCH
    else:
        train_kw["max_epochs"] = UCI_EPOCHS
        train_kw["batch_size"] = min(int(train_kw.get("batch_size", 64)), 128)
    return train_kw


def hpo_c0r(domain: str, n_trials: int) -> dict:
    import optuna

    ceiling = load_scientific_ceiling(domain)
    fit_ids, _, _ = inner_partitions(domain, SCREEN_FOLD)
    prepared = scale_views(domain, fit_ids)
    study = _make_study(_study_name("c0r_sci", domain, CANDIDATE))

    def objective(trial):
        wait_if_hot()
        cfg, train_kw = sample_hybrid(trial, CANDIDATE, prepared.static_dim, prepared.temporal_dim, prepared.aggregate_dim)
        train_kw = _force_train_kw(train_kw, domain)
        out = evaluate_hybrid(domain, CANDIDATE, cfg, train_kw, fold=SCREEN_FOLD, seed=SCREEN_SEED, prepared=prepared)
        ap = {s: r["ap"] for s, r in out["valid"]["stages"].items()}
        sel = selection_objective(ap, ceiling, domain, out["parameter_count"])
        trial.set_user_attr("ap", ap)
        trial.set_user_attr("selection", sel)
        _log(f"HPO {domain} AP={ {k: round(v,4) for k,v in ap.items()} } J={sel['J']:.3f}")
        return float(sel["J"])

    n_done = len([t for t in study.trials if t.state.name == "COMPLETE"])
    remain = max(0, n_trials - n_done)
    _log(f"HPO {domain} C0-R remain={remain}/{n_trials}")
    if remain:
        study.optimize(objective, n_trials=remain, catch=(Exception,))
    completed = [t for t in study.trials if t.state.name == "COMPLETE"]
    summary = {
        "domain": domain,
        "candidate": CANDIDATE,
        "n_complete": len(completed),
        "best_value": study.best_value if completed else None,
        "best_params": study.best_params if completed else None,
        "best_user_attrs": dict(study.best_trial.user_attrs) if completed else None,
        "ceiling": ceiling,
        "outer_test_used": False,
        "protocol_hash": protocol_hash(),
        "scientific": True,
    }
    write_json(RUN_DIR / f"c0r_sci_hpo_{domain}.json", summary)
    return summary


def robust_c0r(domain: str, hpo: dict) -> dict:
    ceiling = load_scientific_ceiling(domain)
    bp = dict(hpo.get("best_params") or {})
    if not bp:
        raise RuntimeError(f"NO_HPO_{domain}")
    rows = []
    preds = []
    for fold in (0, 1, 2):
        fit_ids, _, _ = inner_partitions(domain, fold)
        prepared = scale_views(domain, fit_ids)
        for seed in SEEDS_ROBUST:
            wait_if_hot()
            _log(f"robust {domain} C0-R fold={fold} seed={seed}")
            cfg = make_config(
                CANDIDATE,
                prepared.static_dim,
                prepared.temporal_dim,
                prepared.aggregate_dim,
                d_fuse=int(bp.get("d_fuse", 64)),
                cnn_channels=int(bp.get("cnn_channels", 32)),
                bilstm_hidden=int(bp.get("bilstm_hidden", 32)),
                dropout=float(bp.get("dropout", 0.3)),
            )
            train_kw = _force_train_kw(
                {
                    "lr": float(bp.get("lr", 2e-4)),
                    "weight_decay": float(bp.get("weight_decay", 2e-4)),
                    "pos_weight_multiplier": float(bp.get("pos_weight_multiplier", 1.0)),
                    "lambda_rank": float(bp.get("lambda_rank", 0.15)),
                    "lambda_aux": float(bp.get("lambda_aux", 0.25)),
                    "lambda_kd": 0.0,
                    "batch_size": int(bp.get("batch_size", 64)),
                },
                domain,
            )
            t0 = time.time()
            out = evaluate_hybrid(domain, CANDIDATE, cfg, train_kw, fold=fold, seed=seed, prepared=prepared)
            ap = {s: r["ap"] for s, r in out["valid"]["stages"].items()}
            _log(f"  { {k: round(v,4) for k,v in ap.items()} } {time.time()-t0:.0f}s")
            for s, v in ap.items():
                rows.append({"fold": fold, "seed": seed, "stage": s, "ap": float(v), "recall_at_20": out["valid"]["stages"][s].get("recall_at_20")})
            for rec in out["valid"].get("predictions") or []:
                item = dict(rec)
                item.update({"fold": fold, "seed": seed, "domain": domain, "candidate": CANDIDATE})
                preds.append(item)
    df = pd.DataFrame(rows)
    mean = {s: float(df.loc[df.stage == s, "ap"].mean()) for s in df.stage.unique()}
    std = {s: float(df.loc[df.stage == s, "ap"].std()) for s in df.stage.unique()}
    gate = evaluate_development_gate(domain, df, {s: {"ap": ceiling[s]} for s in ceiling})
    payload = {
        "candidate": CANDIDATE,
        "domain": domain,
        "mean": mean,
        "std": std,
        "gate": gate,
        "ceiling": ceiling,
        "hpo": {"best_value": hpo.get("best_value"), "best_params": hpo.get("best_params")},
        "rows": rows,
        "n_runs": int(df.groupby(["fold", "seed"]).ngroups),
        "outer_test_used": False,
        "protocol_hash": protocol_hash(),
        "scientific": True,
        "frozen_at": utc_now(),
        "git_commit": git_commit(),
    }
    write_json(RUN_DIR / f"c0r_sci_robust_{domain}.json", payload)
    if preds:
        pd.DataFrame(preds).to_parquet(OOF_DIR / f"c0r_sci_oof_{domain}.parquet", index=False)
    write_json(RUN_DIR / f"development_gate_{domain}.json", gate)
    return payload


def write_reports(uci: dict | None, oulad: dict | None) -> None:
    ensure_dirs()
    uci = uci or {}
    oulad = oulad or {}
    ug = (uci.get("gate") or {})
    og = (oulad.get("gate") or {})
    combined = bool(ug.get("pass")) and bool(og.get("pass"))
    status = "READY_FOR_DEFENSE_STRICT" if combined else "NOT_READY_FOR_DEFENSE"
    path = REPORT_ROOT / "C0R_SCIENTIFIC.md"
    path.write_text(
        f"""# C0-R scientific evaluation

Topology remains **C0-R** (same Hybrid CNN–BiLSTM family). C4-STAR is archived under `test_lab/` and is **not** the thesis model.

Outer test used: **false**. Serving Hybrid: **unchanged**.

Status: **{status}**

## UCI 3×3 vs CatBoost

Mean AP: `{json.dumps(uci.get("mean"))}`
Gate pass: `{ug.get("pass")}`
Checks: `{json.dumps(ug.get("checks"), indent=2, default=str)}`

## OULAD 3×3 vs v2.1 ceiling (XGB/LR)

Mean AP: `{json.dumps(oulad.get("mean"))}`
Gate pass: `{og.get("pass")}`
Checks: `{json.dumps(og.get("checks"), indent=2, default=str)}`

## Decision

Do not write vượt trội unless both domain gates pass. C4-STAR numbers in test_lab are not this authority.
""",
        encoding="utf-8",
    )
    write_json(
        RUN_DIR / "development_gate.json",
        {"pass": combined, "uci": ug.get("pass"), "oulad": og.get("pass"), "outer_test_used": False, "candidate": CANDIDATE},
    )


def main() -> int:
    _boost()
    ensure_dirs()
    st = _load_state()
    done = set(st.get("completed") or [])
    write_status(
        phase="C0-R scientific",
        completed=sorted(done),
        evidence=[],
        decision="C0-R is the Hybrid under test; C4-STAR is in test_lab",
        next_step="OULAD HPO then 3x3",
        extra=f"gpu={snapshot()}",
    )
    uci_hpo = oulad_hpo = uci_rob = oulad_rob = None
    try:
        if "hpo_oulad" not in done:
            oulad_hpo = hpo_c0r("oulad", OULAD_HPO_TRIALS)
            done.add("hpo_oulad")
            st["completed"] = sorted(done)
            _save_state(st)
        else:
            oulad_hpo = json.loads((RUN_DIR / "c0r_sci_hpo_oulad.json").read_text(encoding="utf-8"))
        if "robust_oulad" not in done:
            oulad_rob = robust_c0r("oulad", oulad_hpo)
            done.add("robust_oulad")
            st["completed"] = sorted(done)
            _save_state(st)
        else:
            p = RUN_DIR / "c0r_sci_robust_oulad.json"
            oulad_rob = json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}
        if "hpo_uci" not in done:
            uci_hpo = hpo_c0r("uci", UCI_HPO_TRIALS)
            done.add("hpo_uci")
            st["completed"] = sorted(done)
            _save_state(st)
        else:
            uci_hpo = json.loads((RUN_DIR / "c0r_sci_hpo_uci.json").read_text(encoding="utf-8"))
        if "robust_uci" not in done:
            uci_rob = robust_c0r("uci", uci_hpo)
            done.add("robust_uci")
            st["completed"] = sorted(done)
            _save_state(st)
        else:
            p = RUN_DIR / "c0r_sci_robust_uci.json"
            uci_rob = json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}
        write_reports(uci_rob, oulad_rob)
        write_status(
            phase="C0-R scientific done",
            completed=sorted(done),
            evidence=[str(RUN_DIR / "c0r_sci_robust_oulad.json"), str(REPORT_ROOT / "C0R_SCIENTIFIC.md")],
            decision="see C0R_SCIENTIFIC.md — no authority promotion unless both gates pass",
            next_step="read reports",
        )
    except Exception:
        traceback.print_exc()
        write_status(phase="C0-R scientific FAILED", completed=sorted(done), evidence=[], decision="inspect traceback", next_step="resume", blockers=["c0r_scientific"])
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
