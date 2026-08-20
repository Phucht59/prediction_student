"""CLI: audit, prepare, baselines, diagnose, optimize, confirm, report, overnight."""
from __future__ import annotations

import argparse
import json
import traceback
from pathlib import Path

from . import PROTOCOL_VERSION
from .io_utils import git_branch, git_commit, git_dirty, write_json
from .paths import MANIFEST_DIR, PROJECT_ROOT, REPORT_ROOT, RUN_DIR, ensure_dirs
from .protocol import AUTHORITY_REF_COMMIT, CANDIDATES, PROTOCOL_ID, protocol_hash, protocol_payload
from .status import write_status


def cmd_audit(_args) -> int:
    ensure_dirs()
    from .db import health, migrate
    from .hardware import capture_hardware

    hw = capture_hardware(full_bench=False)
    db_state = {"ok": False}
    try:
        db_state = migrate()
        db_state.update(health())
    except Exception as exc:
        db_state = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
    payload = {
        "protocol_id": PROTOCOL_ID,
        "protocol_hash": protocol_hash(),
        "git_commit": git_commit(),
        "git_branch": git_branch(),
        "git_dirty": git_dirty(),
        "authority_ref_commit": AUTHORITY_REF_COMMIT,
        "head_matches_audit_commit": git_commit() == AUTHORITY_REF_COMMIT,
        "hardware": hw,
        "database": db_state,
    }
    write_json(MANIFEST_DIR / "audit.json", payload)
    write_status(
        phase="P0/P1 audit",
        completed=["git/hardware capture", "protocol hash", "DB migrate attempted"],
        evidence=[f"protocol={protocol_hash()[:12]}", f"db_ok={db_state.get('ok')}", f"cuda={hw['benchmark']['cuda']['gpu_name']}"],
        decision="Continue to prepare; do not touch current thesis authority.",
        next_step="python -m experiments.hybrid_superiority_v2 prepare --dataset all",
        blockers=[] if db_state.get("ok") else ["PostgreSQL migrate/health failed; HPO will fall back to in-memory Optuna"],
    )
    print(json.dumps({"protocol_hash": protocol_hash(), "cuda": hw["benchmark"]["cuda"]["gpu_name"], "db": db_state.get("ok")}, indent=2))
    return 0 if hw["benchmark"]["cuda"]["cuda_available"] else 2


def cmd_prepare(args) -> int:
    from .data import make_splits, prepare_all, prepare_oulad, prepare_uci, verify_raw_checksums

    verify_raw_checksums()
    if args.dataset == "all":
        print("prepare all", prepare_all().keys())
    elif args.dataset == "uci":
        print("prepare uci", prepare_uci())
        print("splits uci", make_splits("uci"))
    else:
        print("prepare oulad", prepare_oulad().get("n_enrollments"))
        print("splits oulad", make_splits("oulad"))
    write_status(
        phase="P2 prepare",
        completed=["raw checksums", f"dataset={args.dataset}"],
        evidence=[str(MANIFEST_DIR / "data_lock.json")],
        decision="Data contract built from raw in-repo files; no C:\\hufit\\kltn.",
        next_step="python -m experiments.hybrid_superiority_v2 baselines --resume",
    )
    return 0


def cmd_baselines(args) -> int:
    from .hpo import run_baselines_domain

    domains = ("uci", "oulad") if args.dataset == "all" else (args.dataset,)
    for domain in domains:
        print("baselines", domain)
        lock = run_baselines_domain(domain)
        print(json.dumps(lock["stage_best_ap"], indent=2))
    write_status(
        phase="P4 baseline ceiling",
        completed=[f"baseline lock {d}" for d in domains],
        evidence=[str(RUN_DIR / f"baseline_lock_{d}.json") for d in domains],
        decision="Baseline vector frozen before Hybrid HPO. XGB and CatBoost are in the roster.",
        next_step="python -m experiments.hybrid_superiority_v2 diagnose --candidate C0-R",
    )
    return 0


def cmd_diagnose(args) -> int:
    from .diagnose import diagnose_candidate

    domains = ("uci", "oulad") if args.dataset == "all" else (args.dataset,)
    for domain in domains:
        print("diagnose", domain, args.candidate)
        out = diagnose_candidate(domain, args.candidate)
        print(json.dumps({k: out[k] for k in ("parameter_count", "valid_ap", "full_minus_shuffle", "availability_pass") if k in out}, indent=2, default=str))
    return 0


def cmd_optimize(args) -> int:
    from .hpo import run_hybrid_screen

    domains = ("uci", "oulad") if args.dataset == "all" else (args.dataset,)
    cands = CANDIDATES if args.candidate == "all" else (args.candidate,)
    for domain in domains:
        for cand in cands:
            print("optimize", domain, cand)
            print(json.dumps(run_hybrid_screen(domain, cand), indent=2, default=str))
    return 0


def cmd_confirm(args) -> int:
    gate = RUN_DIR / "development_gate.json"
    if not gate.exists():
        print("CONFIRM_REFUSED: development gate artifact missing")
        return 3
    payload = json.loads(gate.read_text(encoding="utf-8"))
    if not payload.get("pass"):
        print("CONFIRM_REFUSED: development gate failed")
        return 3
    if args.frozen_protocol and args.frozen_protocol != protocol_hash():
        print("CONFIRM_REFUSED: protocol hash mismatch")
        return 3
    print("CONFIRM_NOT_OPENED: nested confirmation runs only after explicit freeze; development still in progress")
    return 0


def cmd_report(_args) -> int:
    from .report import write_all_reports

    write_all_reports()
    print("reports written under", REPORT_ROOT)
    return 0


def cmd_overnight(args) -> int:
    failures = []

    def step(name, fn):
        print("=" * 60, name)
        try:
            fn()
            return True
        except Exception as exc:
            failures.append(f"{name}: {type(exc).__name__}: {exc}")
            traceback.print_exc()
            write_status(
                phase=name,
                completed=[],
                evidence=[],
                decision="recoverable failure; continuing remaining phases",
                next_step="see blockers",
                blockers=failures,
            )
            return False

    step("audit", lambda: cmd_audit(args))
    step("prepare", lambda: cmd_prepare(argparse.Namespace(dataset="all")))
    import pytest
    import sys

    def _tests():
        rc = pytest.main(["-q", str(PROJECT_ROOT / "tests" / "research" / "hybrid_superiority_v2")])
        if rc not in {0, pytest.ExitCode.OK}:
            raise RuntimeError(f"pytest_exit_{rc}")

    step("integrity_tests", _tests)
    step("baselines_uci", lambda: cmd_baselines(argparse.Namespace(dataset="uci", resume=True)))
    step("diagnose_uci", lambda: cmd_diagnose(argparse.Namespace(dataset="uci", candidate="C0-R")))
    step("screen_uci", lambda: cmd_optimize(argparse.Namespace(dataset="uci", candidate="all")))
    step("baselines_oulad", lambda: cmd_baselines(argparse.Namespace(dataset="oulad", resume=True)))
    step("diagnose_oulad", lambda: cmd_diagnose(argparse.Namespace(dataset="oulad", candidate="C0-R")))
    step("screen_oulad", lambda: cmd_optimize(argparse.Namespace(dataset="oulad", candidate=args.candidate or "C3-G")))
    step("report", lambda: cmd_report(args))
    write_status(
        phase="overnight loop",
        completed=["see above"],
        evidence=[],
        decision="continue until development gate or resource exhaustion",
        next_step="inspect reports/research/hybrid_superiority_v2/FINAL_DECISION.md",
        blockers=failures,
    )
    cmd_report(args)
    return 0 if not failures else 1


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="python -m experiments.hybrid_superiority_v2")
    p.add_argument("--dry-run", action="store_true")
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("audit")
    pr = sub.add_parser("prepare")
    pr.add_argument("--dataset", default="all", choices=["all", "uci", "oulad"])
    b = sub.add_parser("baselines")
    b.add_argument("--dataset", default="all", choices=["all", "uci", "oulad"])
    b.add_argument("--resume", action="store_true")
    d = sub.add_parser("diagnose")
    d.add_argument("--dataset", default="all", choices=["all", "uci", "oulad"])
    d.add_argument("--candidate", default="C0-R")
    o = sub.add_parser("optimize")
    o.add_argument("--dataset", default="all", choices=["all", "uci", "oulad"])
    o.add_argument("--candidate", default="C3-G")
    o.add_argument("--resume", action="store_true")
    c = sub.add_parser("confirm")
    c.add_argument("--frozen-protocol", default="")
    sub.add_parser("report")
    ov = sub.add_parser("overnight")
    ov.add_argument("--candidate", default="C3-G")
    ov.add_argument("--resume", action="store_true")
    return p


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.dry_run:
        print({"cmd": args.cmd, "protocol": PROTOCOL_ID, "hash": protocol_hash()})
        return 0
    dispatch = {
        "audit": cmd_audit,
        "prepare": cmd_prepare,
        "baselines": cmd_baselines,
        "diagnose": cmd_diagnose,
        "optimize": cmd_optimize,
        "confirm": cmd_confirm,
        "report": cmd_report,
        "overnight": cmd_overnight,
    }
    return dispatch[args.cmd](args)
