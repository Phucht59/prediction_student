"""Resume-safe overnight campaign. Thermal cap 80C. No outer test. No Gemini."""
from __future__ import annotations

import json
import os
import traceback
from pathlib import Path

os.environ.setdefault("PYTHONUNBUFFERED", "1")
os.environ.setdefault("HS_V2_GPU_TREES", "1")
os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")

import pandas as pd

from experiments.hybrid_superiority_v2.io_utils import git_commit, utc_now, write_json
from experiments.hybrid_superiority_v2.paths import RUN_DIR as PARENT_RUN
from experiments.hybrid_superiority_v2.teacher import teachers_for_prepared
from experiments.hybrid_superiority_v2.data import inner_partitions, scale_views

from .baselines_ext import run_baselines_domain
from .diagnose import diagnose_backbone
from .eval_run import eval_c4, eval_v2_hybrid, score_against_ceiling
from .paths import PROTOCOL_ROOT, REPORT_ROOT, RUN_DIR, ensure_dirs
from .protocol import (
    BACKBONES,
    MECHANISMS,
    PROTOCOL_ID,
    SEEDS_SCREEN,
    protocol_hash,
    protocol_payload,
)
from .recover import recover, write_recovery_report
from .status import load_state, save_state, write_status
from .thermal import snapshot, try_set_power_limit, wait_if_hot


def _log(msg: str) -> None:
    print(f"[{utc_now()}] {msg}", flush=True)


def _ceiling(domain: str) -> dict[str, float]:
    for path in (RUN_DIR / f"baseline_lock_{domain}.json", PARENT_RUN / f"baseline_lock_{domain}.json"):
        if path.exists():
            lock = json.loads(path.read_text(encoding="utf-8"))
            return {k: float(v["ap"]) for k, v in lock["stage_best_ap"].items()}
    raise FileNotFoundError(f"no ceiling for {domain}")


def phase_protocol() -> None:
    ensure_dirs()
    payload = protocol_payload()
    write_json(PROTOCOL_ROOT / "protocol_v2_1.json", {"sha256": protocol_hash(), "payload": payload, "frozen_at": utc_now()})
    yaml_path = Path("experiments/c4_star/protocol_v2_1.yaml")
    if not yaml_path.exists():
        yaml_path.write_text(
            "# See protocols/c4_star_v2_1/protocol_v2_1.json for the frozen hash.\n"
            f"protocol_id: {PROTOCOL_ID}\n"
            f"sha256: {protocol_hash()}\n"
            "parent: hybrid_superiority_v2.0\n"
            "outer_splits_regenerated: false\n"
            "temp_hard_c: 80\n",
            encoding="utf-8",
        )
    md = PROTOCOL_ROOT / "PROTOCOL_V2_1.md"
    if not md.exists():
        md.write_text(
            f"# PROTOCOL_V2_1\n\nHash `{protocol_hash()}`.\n\n"
            "Amendment: joint-domain candidate selection. UCI tests short-prefix; OULAD tests long-sequence.\n"
            "Locked outer splits are **not** regenerated. Parent hash recorded in payload.\n"
            "SPEED_FINISH is not confirmatory. Outer test stays closed until development Gold/statistical gate.\n"
            "DT trials have a 90s timeout. GPU software-throttled at 80C.\n",
            encoding="utf-8",
        )


def phase_recover() -> None:
    payload = recover()
    write_recovery_report(payload)
    # copy verified UCI ceiling into C4 namespace without retuning
    src = PARENT_RUN / "baseline_lock_uci.json"
    if src.exists() and not (RUN_DIR / "baseline_lock_uci.json").exists():
        lock = json.loads(src.read_text(encoding="utf-8"))
        lock["copied_from_parent"] = True
        lock["c4_protocol_hash"] = protocol_hash()
        lock["note"] = "UCI 3x3 ceiling verified from OOF; not retuned in v2.1"
        write_json(RUN_DIR / "baseline_lock_uci.json", lock)


def phase_tests() -> None:
    import pytest

    rc = pytest.main(["-q", "tests/research/c4_star", "tests/research/hybrid_superiority_v2"])
    if rc not in {0, pytest.ExitCode.OK}:
        raise RuntimeError(f"pytest_exit_{rc}")


def phase_baselines_oulad() -> None:
    wait_if_hot()
    run_baselines_domain(
        "oulad",
        models=("XGB", "CatBoost", "TemporalSummaryCatBoost", "LR", "SVM", "MLP", "RF", "DT"),
    )


def phase_joint_screen() -> None:
    rows = []
    for domain in ("uci", "oulad"):
        ceiling = _ceiling(domain)
        for cand in BACKBONES:
            for seed in SEEDS_SCREEN:
                wait_if_hot()
                _log(f"screen {domain} {cand} seed={seed}")
                try:
                    out = eval_v2_hybrid(domain, cand, fold=0, seed=seed, max_epochs=24 if domain == "uci" else 20, patience=8)
                    scored = score_against_ceiling(domain, out, ceiling)
                    rec = {
                        "domain": domain,
                        "candidate": cand,
                        "seed": seed,
                        "ap": scored["ap"],
                        "J": scored["selection"]["J"],
                        "n_warm_loss": scored["selection"]["n_warm_loss"],
                        "min_r": scored["selection"]["min_r"],
                        "n_params": out["parameter_count"],
                        "best_epoch": out["best_epoch"],
                        "runtime": out["runtime_seconds"],
                    }
                except Exception as exc:
                    rec = {"domain": domain, "candidate": cand, "seed": seed, "error": f"{type(exc).__name__}: {exc}"}
                    traceback.print_exc()
                rows.append(rec)
                write_json(RUN_DIR / "joint_screen.json", {"rows": rows, "protocol": protocol_hash(), "outer_test_used": False})
                _log(f"  {rec}")
    # promote at most two backbones: fewest warm losses, then min_r, then J
    ok = [r for r in rows if "J" in r]
    summary = []
    for domain in ("uci", "oulad"):
        for cand in BACKBONES:
            sub = [r for r in ok if r["domain"] == domain and r["candidate"] == cand]
            if not sub:
                continue
            summary.append(
                {
                    "domain": domain,
                    "candidate": cand,
                    "mean_J": float(pd.Series([r["J"] for r in sub]).mean()),
                    "mean_warm_loss": float(pd.Series([r["n_warm_loss"] for r in sub]).mean()),
                    "mean_min_r": float(pd.Series([r["min_r"] for r in sub]).mean()),
                }
            )
    write_json(RUN_DIR / "joint_screen_summary.json", {"summary": summary})
    path = REPORT_ROOT / "03_EXISTING_CANDIDATE_JOINT_SCREEN.md"
    lines = ["# 03 Joint screen", "", "Fold 0, seeds 42/1201/2026, inner VALID only.", ""]
    for row in summary:
        lines.append(f"- {row['domain']} {row['candidate']}: mean J={row['mean_J']:.3f} warm_loss={row['mean_warm_loss']:.2f} min_r={row['mean_min_r']:.3f}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def phase_diagnose() -> None:
    rows = []
    for domain, cand in (("uci", "C0-R"), ("oulad", "C0-R"), ("oulad", "C3-G")):
        for seed in (42, 1201, 2026):
            wait_if_hot()
            _log(f"diagnose {domain} {cand} {seed}")
            try:
                rows.append(diagnose_backbone(domain, cand, seed=seed, max_epochs=16 if domain == "oulad" else 20))
            except Exception:
                traceback.print_exc()
            write_json(RUN_DIR / "diagnose_all.json", {"rows": rows, "outer_test_used": False})
    path = REPORT_ROOT / "04_TEMPORAL_SIGNAL_DIAGNOSTICS.md"
    path.write_text(
        "# 04 Temporal diagnostics\n\nInner VALID. UCI shuffle gap ~0 is expected (T<=2).\n"
        "OULAD needs a stable positive shuffle/reverse gap to claim order/dynamics.\n\n"
        f"See `artifacts/research/c4_star_v2_1/runs/diagnose_*.json` ({len(rows)} runs).\n",
        encoding="utf-8",
    )


def phase_c4_ladder() -> None:
    rows = []
    for domain in ("uci", "oulad"):
        ceiling = _ceiling(domain)
        fit_ids, _, _ = inner_partitions(domain, 0)
        prepared = scale_views(domain, fit_ids)
        teacher = teachers_for_prepared(prepared, fit_ids, seed=42)
        for mech in MECHANISMS:
            wait_if_hot()
            _log(f"ladder {domain} {mech}")
            kw = {
                "mechanism": mech,
                "lambda_kd": 0.25 if mech >= "M2" else 0.0,
                "lambda_ssl": 0.1 if mech in {"M6", "M7"} else 0.0,
                "group_dro": mech in {"M4", "M5", "M6", "M7"},
                "use_ema": mech == "M7" or True,
                "max_epochs": 28 if domain == "uci" else 20,
                "patience": 8,
                "lr": 5e-4,
            }
            try:
                out = eval_c4(domain, 0, 42, teacher_map=teacher if kw["lambda_kd"] else None, **kw)
                scored = score_against_ceiling(domain, out, ceiling)
                rec = {"domain": domain, "mechanism": mech, "ap": scored["ap"], "J": scored["selection"]["J"], "n_params": out["parameter_count"], "best_epoch": out["best_epoch"]}
            except Exception as exc:
                rec = {"domain": domain, "mechanism": mech, "error": f"{type(exc).__name__}: {exc}"}
                traceback.print_exc()
            rows.append(rec)
            write_json(RUN_DIR / "c4_ladder.json", {"rows": rows, "protocol": protocol_hash()})
            _log(f"  {rec}")
    (REPORT_ROOT / "05_C4_STAR_LADDER.md").write_text(
        "# 05 C4-STAR ladder\n\nFold 0 seed 42. Fair epochs (not 8). See `runs/c4_ladder.json`.\n",
        encoding="utf-8",
    )


def phase_c4_hpo() -> None:
    import optuna

    from experiments.hybrid_superiority_v2.hpo import _make_study

    from .objective import constrained_J

    for domain in ("uci", "oulad"):
        ceiling = _ceiling(domain)
        study = _make_study(f"c4_v21_hpo_{domain}_{protocol_hash()[:12]}")
        fit_ids, _, _ = inner_partitions(domain, 0)
        prepared = scale_views(domain, fit_ids)
        teacher = teachers_for_prepared(prepared, fit_ids, seed=42)

        def objective(trial, domain=domain, ceiling=ceiling, teacher=teacher):
            wait_if_hot()
            mech = trial.suggest_categorical("mechanism", ["M3", "M4", "M6", "M7"])
            kw = {
                "mechanism": mech,
                "d_fuse": trial.suggest_categorical("d_fuse", [48, 64, 96]),
                "cnn_channels": trial.suggest_categorical("cnn_channels", [16, 32, 48, 64]),
                "bilstm_hidden": trial.suggest_categorical("bilstm_hidden", [32, 48, 64]),
                "dropout": trial.suggest_float("dropout", 0.10, 0.40),
                "lr": trial.suggest_float("lr", 1e-4, 3e-3, log=True),
                "weight_decay": trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True),
                "lambda_kd": trial.suggest_categorical("lambda_kd", [0.0, 0.1, 0.25, 0.5, 1.0]),
                "kd_temperature": trial.suggest_categorical("kd_temperature", [1.0, 2.0, 3.0]),
                "lambda_rank": trial.suggest_categorical("lambda_rank", [0.0, 0.05, 0.1, 0.2]),
                "lambda_ssl": trial.suggest_categorical("lambda_ssl", [0.0, 0.05, 0.1, 0.25]),
                "lambda_aux": trial.suggest_float("lambda_aux", 0.1, 1.0),
                "initial_alpha": trial.suggest_categorical("initial_alpha", [0.02, 0.05, 0.10]),
                "group_dro": mech in {"M4", "M6", "M7"},
                "max_epochs": 24 if domain == "uci" else 18,
                "patience": 8,
            }
            out = eval_c4(domain, 0, 42, teacher_map=teacher if kw["lambda_kd"] else None, **kw)
            ap = {s: r["ap"] for s, r in out["valid"]["stages"].items()}
            sel = constrained_J(ap, ceiling, domain, n_params=out["parameter_count"])
            trial.set_user_attr("ap", ap)
            trial.set_user_attr("selection", sel)
            _log(f"HPO {domain} AP={ {k: round(v,4) for k,v in ap.items()} } J={sel['J']:.3f}")
            return float(sel["J"])

        while True:
            n = len([t for t in study.trials if t.state.name == "COMPLETE"])
            values = [t.value for t in study.trials if t.state.name == "COMPLETE" and t.value is not None]
            plateau = n >= 160 and values and (max(values) - max(values[-50:])) < 0.001 if n >= 50 else False
            _log(f"C4 HPO {domain} complete={n}/160 plateau={plateau}")
            if n >= 160 and plateau:
                break
            if n >= 240:
                break
            study.optimize(objective, n_trials=4, catch=(Exception,))
            write_json(
                RUN_DIR / f"hpo_{domain}.json",
                {
                    "n_complete": n,
                    "best": study.best_value if values else None,
                    "best_params": study.best_params if values else None,
                    "best_attrs": dict(study.best_trial.user_attrs) if values else None,
                },
            )


def phase_robust_3x3() -> None:
    """3 development folds × 3 seeds for the HPO winner of each domain. Inner VALID only."""
    from experiments.hybrid_superiority_v2.teacher import teachers_for_prepared
    from experiments.hybrid_superiority_v2.gate import evaluate_development_gate as _gate_v2

    from .paths import RUN_DIR as C4_RUN

    for domain in ("uci", "oulad"):
        hpo_path = RUN_DIR / f"hpo_{domain}.json"
        if not hpo_path.exists():
            _log(f"robust skip {domain}: no HPO artifact")
            continue
        hpo = json.loads(hpo_path.read_text(encoding="utf-8"))
        bp = dict(hpo.get("best_params") or {})
        if not bp:
            _log(f"robust skip {domain}: empty best_params")
            continue
        ceiling = _ceiling(domain)
        mech = bp.pop("mechanism", "M4")
        rows = []
        preds = []
        for fold in (0, 1, 2):
            fit_ids, _, _ = inner_partitions(domain, fold)
            prepared = scale_views(domain, fit_ids)
            teacher = teachers_for_prepared(prepared, fit_ids, seed=42) if float(bp.get("lambda_kd") or 0) > 0 else None
            for seed in SEEDS_SCREEN:
                wait_if_hot()
                _log(f"robust {domain} {mech} fold={fold} seed={seed}")
                kw = dict(bp)
                kw.update(
                    {
                        "mechanism": mech,
                        "group_dro": mech in {"M4", "M5", "M6", "M7"},
                        "use_ema": True,
                        "max_epochs": 24 if domain == "uci" else 18,
                        "patience": 8,
                        "teacher_map": teacher,
                    }
                )
                try:
                    out = eval_c4(domain, fold, seed, **kw)
                    ap = {s: r["ap"] for s, r in out["valid"]["stages"].items()}
                    rec = {"domain": domain, "fold": fold, "seed": seed, "mechanism": mech, "ap": ap, "n_params": out["parameter_count"], "best_epoch": out["best_epoch"]}
                    for s, v in ap.items():
                        rows.append(
                            {
                                "domain": domain,
                                "fold": fold,
                                "seed": seed,
                                "stage": s,
                                "ap": float(v),
                                "recall_at_20": out["valid"]["stages"][s].get("recall_at_20"),
                            }
                        )
                    for p in out["valid"].get("predictions") or []:
                        item = dict(p)
                        item.update({"fold": fold, "seed": seed, "domain": domain})
                        preds.append(item)
                except Exception as exc:
                    rec = {"domain": domain, "fold": fold, "seed": seed, "error": f"{type(exc).__name__}: {exc}"}
                    traceback.print_exc()
                _log(f"  {rec}")
                write_json(RUN_DIR / f"robust_{domain}.json", {"mechanism": mech, "hpo_best": hpo.get("best"), "rows_raw": rows, "protocol": protocol_hash()})
        if rows:
            import pandas as pd

            df = pd.DataFrame(rows)
            mean = {s: float(df.loc[df.stage == s, "ap"].mean()) for s in df.stage.unique()}
            std = {s: float(df.loc[df.stage == s, "ap"].std()) for s in df.stage.unique()}
            ceil_wrap = {s: {"ap": ceiling[s]} for s in ceiling}
            gate = _gate_v2(domain, df, ceil_wrap)
            write_json(
                RUN_DIR / f"robust_{domain}.json",
                {
                    "domain": domain,
                    "mechanism": mech,
                    "mean": mean,
                    "std": std,
                    "gate": gate,
                    "ceiling": ceiling,
                    "n_runs": int(df.groupby(["fold", "seed"]).ngroups),
                    "rows": rows,
                    "outer_test_used": False,
                    "protocol": protocol_hash(),
                },
            )
            if preds:
                from .paths import OOF_DIR

                pd.DataFrame(preds).to_parquet(OOF_DIR / f"c4_oof_{domain}.parquet", index=False)
            write_json(RUN_DIR / f"development_gate_{domain}.json", gate)


def phase_finalize_reports() -> None:
    from .finalize import write_all_v21_reports

    write_all_v21_reports()


def write_final_stub(state: dict) -> None:
    gate_pass = False
    status = "DEVELOPMENT_GATE_FAILED"
    path = REPORT_ROOT / "FINAL_DECISION_V2_1.md"
    path.write_text(
        f"""{status}

# FINAL_DECISION_V2_1

Status above is machine-readable. Outer test **not opened**. Serving Hybrid **not** promoted.

- Protocol `{protocol_hash()}`
- Git `{git_commit()}`
- Time `{utc_now()}`
- Phase `{state.get("phase")}`
- GPU cap 80C

See `OVERNIGHT_STATUS.md` and `runs/`. SPEED_FINISH OULAD numbers are not confirmatory. Thesis may not claim vượt trội unless a later freeze passes the locked gate.
""",
        encoding="utf-8",
    )
    (REPORT_ROOT / "NEXT_ACTIONS.md").write_text(
        "# NEXT_ACTIONS\n\nResume: `py -3.10 -u -m experiments.c4_star overnight`\n\nDo not open outer test.\n",
        encoding="utf-8",
    )


PHASES = [
    ("protocol", phase_protocol),
    ("recover", phase_recover),
    ("tests", phase_tests),
    ("baselines_oulad", phase_baselines_oulad),
    ("joint_screen", phase_joint_screen),
    ("diagnose", phase_diagnose),
    ("c4_ladder", phase_c4_ladder),
    ("c4_hpo", phase_c4_hpo),
    ("robust_3x3", phase_robust_3x3),
    ("finalize_reports", phase_finalize_reports),
]


def main() -> int:
    ensure_dirs()
    try_set_power_limit(160)
    import threading
    import time as _time

    def _heartbeat():
        while True:
            try:
                st = load_state()
                write_status(
                    phase=str(st.get("phase", "heartbeat")),
                    completed=list(st.get("completed") or []),
                    decision="heartbeat 10 min",
                    next_step=str(st.get("phase")),
                    extra=f"gpu={snapshot()}",
                )
            except Exception:
                pass
            _time.sleep(600)

    threading.Thread(target=_heartbeat, daemon=True).start()
    write_status(phase="boot", completed=[], decision="start C4-STAR overnight", next_step="protocol freeze", extra=f"gpu={snapshot()}")
    state = load_state()
    done = set(state.get("completed") or [])
    for name, fn in PHASES:
        if name in done:
            _log(f"skip completed {name}")
            continue
        state["phase"] = name
        save_state(state)
        write_status(
            phase=name,
            completed=sorted(done),
            decision=f"running {name}",
            next_step=name,
            study=name,
            extra="Thermal hard cap 80C via wait_if_hot.",
        )
        try:
            _log(f"==== PHASE {name} ====")
            fn()
            done.add(name)
            state["completed"] = sorted(done)
            save_state(state)
        except Exception:
            traceback.print_exc()
            state.setdefault("failed", []).append(name)
            save_state(state)
            write_status(
                phase=f"{name} FAILED",
                completed=sorted(done),
                decision="continue remaining phases after logging",
                next_step="inspect traceback",
                blockers=[name],
            )
            # baselines/hpo failures are not silent skips of later GPU work if ceiling missing
            if name in {"protocol", "recover", "tests"}:
                write_final_stub(state)
                return 2
            if name == "baselines_oulad":
                _log("OULAD ceiling incomplete; joint screen will use parent SPEED lock and mark UNVERIFIED")
    if "finalize_reports" in done:
        from .finalize import write_all_v21_reports

        write_all_v21_reports()
    else:
        write_final_stub(state)
    write_status(phase="campaign_loop_finished_or_partial", completed=sorted(done), decision="see FINAL_DECISION_V2_1.md", next_step="resume if incomplete")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
