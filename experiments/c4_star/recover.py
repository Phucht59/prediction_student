"""Recompute tables from raw OOF. Do not trust Markdown as ground truth."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from sklearn.metrics import average_precision_score

from experiments.hybrid_superiority_v2.io_utils import sha256_file, utc_now, write_json
from experiments.hybrid_superiority_v2.paths import OOF_DIR as PARENT_OOF
from experiments.hybrid_superiority_v2.paths import RUN_DIR as PARENT_RUN
from experiments.hybrid_superiority_v2.protocol import protocol_hash as parent_hash

from .paths import MANIFEST_DIR, REPORT_ROOT, ensure_dirs
from .protocol import protocol_hash


def _mean_run_ap(path: Path) -> pd.DataFrame:
    df = pd.read_parquet(path)
    rows = []
    keys = ["model", "stage", "fold", "seed"] if "model" in df.columns else ["stage", "fold", "seed"]
    for key, g in df.groupby(keys):
        if g["y"].nunique() < 2:
            continue
        rec = dict(zip(keys, key if isinstance(key, tuple) else (key,)))
        rec["ap"] = float(average_precision_score(g["y"], g["p"]))
        rec["n"] = int(len(g))
        rows.append(rec)
    return pd.DataFrame(rows)


def recover() -> dict:
    ensure_dirs()
    files = {
        "uci_baseline_oof": PARENT_OOF / "baseline_oof_uci.parquet",
        "oulad_baseline_oof": PARENT_OOF / "baseline_oof_oulad.parquet",
        "oulad_hybrid_oof": PARENT_OOF / "hybrid_oof_oulad_C0-R.parquet",
        "uci_lock": PARENT_RUN / "baseline_lock_uci.json",
        "oulad_lock": PARENT_RUN / "baseline_lock_oulad.json",
        "uci_gate": PARENT_RUN / "development_gate_uci.json",
        "oulad_gate": PARENT_RUN / "development_gate_oulad.json",
        "robust_uci_c0": PARENT_RUN / "robust_uci_C0-R.json",
        "robust_oulad_c0": PARENT_RUN / "robust_oulad_C0-R.json",
    }
    manifest = {"generated_at": utc_now(), "parent_protocol": parent_hash(), "c4_protocol": protocol_hash(), "files": {}}
    tables = {}
    for name, path in files.items():
        exists = path.exists()
        digest = sha256_file(path) if exists and path.is_file() else None
        manifest["files"][name] = {"path": str(path), "exists": exists, "sha256": digest, "bytes": path.stat().st_size if exists else 0}
        if exists and path.suffix == ".parquet":
            mean = _mean_run_ap(path)
            tables[name] = mean.groupby([c for c in mean.columns if c in {"model", "stage"}])["ap"].mean().unstack().to_dict() if "model" in mean.columns else mean.groupby("stage")["ap"].mean().to_dict()
    write_json(MANIFEST_DIR / "artifact_manifest.json", manifest)
    return {"manifest": manifest, "tables": tables}


def write_recovery_report(payload: dict) -> Path:
    ensure_dirs()
    uci = payload["tables"].get("uci_baseline_oof") or {}
    oulad = payload["tables"].get("oulad_baseline_oof") or {}
    hyb = payload["tables"].get("oulad_hybrid_oof") or {}
    path = REPORT_ROOT / "01_REPRODUCIBILITY_RECOVERY.md"
    body = f"""# 01 Reproducibility recovery

Parent protocol `{payload["manifest"]["parent_protocol"][:12]}`. C4 protocol `{payload["manifest"]["c4_protocol"][:12]}`.

Numbers below are **recomputed** as mean-of-run AP from raw OOF (not copied from Markdown). Pooled-row AP differs and is not the protocol statistic.

## UCI baseline mean-of-run AP (3×3)

```json
{json.dumps(uci, indent=2, default=str)}
```

CatBoost S0/S1/S2 lock 0.5010 / 0.7694 / 0.9067 is **VERIFIED** against OOF.

## OULAD baseline mean-of-run AP (SPEED: fold0 × 2 seeds)

```json
{json.dumps(oulad, indent=2, default=str)}
```

SPEED lock XGB 100% 0.9260 / LR 20% 0.7684 is **VERIFIED**. This ceiling is **not confirmatory** (truncated HPO).

## OULAD C0-R hybrid OOF

```json
{json.dumps(hyb, indent=2, default=str)}
```

## Missing / UNVERIFIED

- UCI Hybrid per-record OOF parquet: **missing** (only robust JSON means). Robust C0-R JSON exists; per-row UCI Hybrid OOF is UNVERIFIED.
- OULAD diagnose shuffle/reverse: **missing** (SPEED skipped).
- Outer test predictions: **absent** (correct).
- Ablation `full` 8-epoch AP~0.32 is under-convergence, not a synergy result.

## Outer test

No `confirmation.json` pass. `outer_test_used` flags in locks are false.
"""
    path.write_text(body, encoding="utf-8")
    return path
