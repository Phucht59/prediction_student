"""P0.1 ablation. Same inner folds as lock. Does not open outer test."""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from research.kltn_science_fix.data import (
    build_uci_grade_views,
    copy_locked_splits,
    inner_partitions,
    load_phase2_cache,
    scale_prepared,
)
from research.kltn_science_fix.model_ablation import ABLATION_ARMS, GRADE_ARMS
from research.kltn_science_fix.paths import ART, REP, RUN, ensure
from research.kltn_science_fix.train import StageTrainer

FOLDS = (0, 1, 2)
SEEDS = (42, 1201, 2026)


def jobs() -> list[dict]:
    out = []
    for ablation in ABLATION_ARMS:
        for fold in FOLDS:
            for seed in SEEDS:
                out.append({"domain": "uci", "stage": "S1", "ablation": ablation, "grade_mode": "both", "fold": fold, "seed": seed})
                out.append({"domain": "oulad", "stage": "35pct", "ablation": ablation, "grade_mode": None, "fold": fold, "seed": seed})
    for mode in GRADE_ARMS:
        for fold in FOLDS:
            for seed in SEEDS:
                out.append({"domain": "uci", "stage": "S1", "ablation": "full", "grade_mode": mode, "fold": fold, "seed": seed})
    # extra S0 full vs tabular to interpret ΔAP S0→S1
    for ablation in ("full", "tabular_only"):
        for fold in FOLDS:
            for seed in SEEDS:
                out.append({"domain": "uci", "stage": "S0", "ablation": ablation, "grade_mode": "both", "fold": fold, "seed": seed})
    return out


def run_id(job: dict) -> str:
    grade = job["grade_mode"] or "na"
    return f"{job['domain']}_{job['stage']}_{job['ablation']}_g{grade}_f{job['fold']}_s{job['seed']}"


def main() -> None:
    ensure()
    copy_locked_splits()
    cache: dict[tuple, object] = {}
    rows = []
    planned = jobs()
    print(f"ablation jobs {len(planned)}", flush=True)
    for i, job in enumerate(planned, 1):
        rid = run_id(job)
        path = RUN / f"{rid}.json"
        if path.exists():
            payload = json.loads(path.read_text(encoding="utf-8"))
        else:
            key = (job["domain"], job["grade_mode"], job["fold"])
            if key not in cache:
                if job["domain"] == "uci":
                    views, context = build_uci_grade_views(job["grade_mode"] or "both")
                else:
                    views, context = load_phase2_cache("oulad")
                fit, stop, valid = inner_partitions(job["domain"], context, job["fold"])
                prepared = scale_prepared(job["domain"], views, context, fit)
                cache[key] = (prepared, fit, stop, valid)
            prepared, fit, stop, valid = cache[key]
            trainer = StageTrainer(prepared, job["stage"], job["ablation"], seed=job["seed"])
            print(f"[{i}/{len(planned)}] {rid}", flush=True)
            payload = trainer.fit(fit, stop, valid, rid)
            del trainer
            import torch

            torch.cuda.empty_cache()
        payload["inner_fold"] = job["fold"]
        payload["grade_mode"] = job["grade_mode"]
        rows.append(payload)
        print(
            f"done {rid} AP={payload.get('valid', {}).get('ap')} sec={payload.get('seconds')}",
            flush=True,
        )
    csv_path = ART / "ablation_raw.csv"
    fields = [
        "run_id",
        "domain",
        "stage",
        "ablation",
        "grade_mode",
        "inner_fold",
        "seed",
        "ap",
        "roc_auc",
        "precision",
        "recall",
        "f1",
        "accuracy",
        "ece",
        "threshold",
        "n_valid",
        "tabular_mass",
        "cnn_mass",
        "bilstm_mass",
        "seconds",
        "outer_test_used",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for payload in rows:
            v = payload.get("valid") or {}
            g = payload.get("gate") or {}
            writer.writerow(
                {
                    "run_id": payload.get("run_id"),
                    "domain": payload.get("domain"),
                    "stage": payload.get("stage"),
                    "ablation": payload.get("ablation"),
                    "grade_mode": payload.get("grade_mode"),
                    "inner_fold": payload.get("inner_fold"),
                    "seed": payload.get("seed"),
                    "ap": v.get("ap"),
                    "roc_auc": v.get("roc_auc"),
                    "precision": v.get("precision"),
                    "recall": v.get("recall"),
                    "f1": v.get("f1"),
                    "accuracy": v.get("accuracy"),
                    "ece": v.get("ece"),
                    "threshold": payload.get("threshold"),
                    "n_valid": v.get("n"),
                    "tabular_mass": g.get("tabular_mass"),
                    "cnn_mass": g.get("cnn_mass"),
                    "bilstm_mass": g.get("bilstm_mass"),
                    "seconds": payload.get("seconds"),
                    "outer_test_used": False,
                }
            )
    print("wrote", csv_path, flush=True)
    _write_ablation_md(csv_path)


def _write_ablation_md(csv_path: Path) -> None:
    import pandas as pd

    frame = pd.read_csv(csv_path)
    lines = [
        "# ABLATION",
        "",
        "Protocol: inner 3 fold × 3 seed, outer fold 0 firewall, FIT-only scale/`pos_weight`, STOP early-stop AP, threshold F1→recall→|t−0.5|.",
        "Single-stage training (UCI S1 / OULAD 35%, plus UCI S0 control). Not the mixed-state serving checkpoint.",
        f"Raw CSV: `{csv_path.as_posix()}`.",
        "",
        "## Mean AP ± std (9 run)",
        "",
        "| domain | stage | ablation | grade_mode | AP mean | AP std | n |",
        "|---|---|---|---|---:|---:|---:|",
    ]
    grouped = frame.groupby(["domain", "stage", "ablation", "grade_mode"], dropna=False)
    for key, sub in grouped:
        lines.append(
            f"| {key[0]} | {key[1]} | {key[2]} | {key[3]} | {sub.ap.mean():.4f} | {sub.ap.std(ddof=1) if len(sub)>1 else float('nan'):.4f} | {len(sub)} |"
        )
    lines += [
        "",
        "## How to read",
        "",
        "- CNN/BiLSTM contribution = full − tabular_only (same stage).",
        "- ΔAP S0→S1 on `full` vs `tabular_only` shows whether the jump is G1 appearing or the temporal modules.",
        "- G1/G2 arms (both / temporal_only / aggregate_only) quantify duplicate-grade leakage on UCI S1.",
        "- If concat ≥ full, the softmax gate is not buying AP on this panel.",
        "",
    ]
    (REP / "ABLATION.md").write_text("\n".join(lines), encoding="utf-8")
    print("wrote", REP / "ABLATION.md", flush=True)


if __name__ == "__main__":
    main()
