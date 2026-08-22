"""Overnight BoostHybrid 3×3 UCI + OULAD. Resume-safe. Moderate GPU load."""
from __future__ import annotations

import csv
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch

from research.kltn_science_fix.boost.features import add_temporal_deltas
from research.kltn_science_fix.boost.train import BOOST_RUN, MixedTrainer
from research.kltn_science_fix.data import (
    build_uci_grade_views,
    copy_locked_splits,
    inner_partitions,
    load_phase2_cache,
    scale_prepared,
)
from research.kltn_science_fix.paths import ART, REP, ensure

FOLDS = (0, 1, 2)
SEEDS = (42, 1201, 2026)
LOCKED = {
    ("uci", "S1"): {"ap": 0.8214, "f1": 0.6899},
    ("uci", "S2"): {"ap": 0.9101, "f1": 0.8010},
    ("oulad", "35pct"): {"ap": 0.8058, "f1": 0.7001},
    ("oulad", "50pct"): {"ap": 0.8483, "f1": 0.7306},
    ("oulad", "75pct"): {"ap": 0.8885, "f1": 0.7807},
}


def jobs() -> list[dict]:
    out = []
    for domain in ("uci", "oulad"):
        for fold in FOLDS:
            for seed in SEEDS:
                out.append({"domain": domain, "fold": fold, "seed": seed})
    return out


def run_id(job: dict) -> str:
    return f"boost_{job['domain']}_f{job['fold']}_s{job['seed']}"


def main() -> None:
    ensure()
    copy_locked_splits()
    cache = {}
    planned = jobs()
    print(f"boost jobs {len(planned)} (safe GPU)", flush=True)
    rows = []
    for i, job in enumerate(planned, 1):
        rid = run_id(job)
        path = BOOST_RUN / f"{rid}.json"
        if path.exists():
            payload = json.loads(path.read_text(encoding="utf-8"))
        else:
            key = (job["domain"], job["fold"])
            if key not in cache:
                if job["domain"] == "uci":
                    views, context = build_uci_grade_views("both")
                else:
                    views, context = load_phase2_cache("oulad")
                views = add_temporal_deltas(views)
                fit, stop, valid = inner_partitions(job["domain"], context, job["fold"])
                prepared = scale_prepared(job["domain"], views, context, fit)
                cache[key] = (prepared, fit, stop, valid)
            prepared, fit, stop, valid = cache[key]
            print(f"[{i}/{len(planned)}] {rid}", flush=True)
            trainer = MixedTrainer(prepared, seed=job["seed"])
            payload = trainer.fit(fit, stop, valid, rid)
            del trainer
            torch.cuda.empty_cache()
            time.sleep(2)
        payload["inner_fold"] = job["fold"]
        rows.append(payload)
        print(f"done {rid} score={payload.get('stop_score_best')} sec={payload.get('seconds')}", flush=True)
    _write_report(rows)


def _write_report(rows: list[dict]) -> None:
    csv_path = ART / "boost" / "boost_valid.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["run_id", "domain", "inner_fold", "seed", "stage", "ap", "f1", "precision", "recall", "roc_auc", "ece", "locked_ap"]
    flat = []
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for payload in rows:
            for stage, m in (payload.get("valid") or {}).items():
                rec = {
                    "run_id": payload.get("run_id"),
                    "domain": payload.get("domain"),
                    "inner_fold": payload.get("inner_fold"),
                    "seed": payload.get("seed"),
                    "stage": stage,
                    "ap": m.get("ap"),
                    "f1": m.get("f1"),
                    "precision": m.get("precision"),
                    "recall": m.get("recall"),
                    "roc_auc": m.get("roc_auc"),
                    "ece": m.get("ece"),
                    "locked_ap": (LOCKED.get((payload.get("domain"), stage)) or {}).get("ap"),
                }
                writer.writerow(rec)
                flat.append(rec)
    lines = [
        "# BOOST night — Hybrid dùng chung UCI+OULAD",
        "",
        "Nhãn nhị phân **không đổi** (OULAD Fail|Withdrawn). Bỏ train/STOP **20%**. STOP 35/50/75 và UCI S1+S2.",
        "Chỉ số STOP: AP + 0.3 F1 + 0.15 ROC-AUC + 0.1 Rec + 0.1 Prec − 0.25 ECE.",
        "Kernel CNN 3 (cả hai miền), Δ tuần, last-step, FiLM progress, rank loss 0.05.",
        f"CSV: `{csv_path.as_posix()}`.",
        "",
        "| domain | stage | Boost AP | locked AP | Boost F1 | n |",
        "|---|---|---:|---:|---:|---:|",
    ]
    import pandas as pd

    frame = pd.DataFrame(flat)
    if not frame.empty:
        for (domain, stage), sub in frame.groupby(["domain", "stage"]):
            locked = (LOCKED.get((domain, stage)) or {}).get("ap")
            lines.append(
                f"| {domain} | {stage} | {sub.ap.mean():.4f} | {locked if locked is not None else '—'} | {sub.f1.mean():.4f} | {len(sub)} |"
            )
    (REP / "BOOST_NIGHT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("wrote", REP / "BOOST_NIGHT.md", flush=True)


if __name__ == "__main__":
    main()
