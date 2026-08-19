"""Inner-only protocol lock, CUDA fail-fast, and kltn namespace bootstrap."""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[2]
KLTN = Path(r"C:\hufit\kltn")
ART = ROOT / "artifacts" / "hybrid_vnext" / "phase2"
CACHE = ART / "cache"
RUNS = ART / "runs"
REPORTS = ROOT / "reports" / "hybrid_vnext" / "phase2"
AUTHORITY_REF = "codex/backup-hybrid-phase8-2026-08-17"
DEVELOPMENT_OUTER_FOLD = 0
SEEDS = (42, 1201, 2026)
FOLDS = (0, 1, 2)
SCREEN_FOLD = 0
SCREEN_SEED = 42
FORBIDDEN_UCI = ("G3", "absences")
FORBIDDEN_OULAD = ("final_result", "target", "score", "date_unregistration")
UCI_STAGES = ("S0", "S1", "S2")
OULAD_PRIMARY = ("20pct", "35pct", "50pct", "75pct")
SPLIT_HASHES_EXPECTED = {
    "uci_inner": "ad8f44e5931d652e353d9d9ebe7b0e840eca3d895243b92d57deb0b3b6e02ae8",
    "oulad_inner": "8559efcf156bcb05eb0a2bdf9e945d54f3989358d8f15064dab1204cd872650c",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def git_commit() -> str:
    try:
        return subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def git_branch() -> str:
    try:
        return subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "--abbrev-ref", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")


def require_cuda() -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA_REQUIRED_FOR_HYBRID_PHASE2")
    props = torch.cuda.get_device_properties(0)
    info = {
        "cuda_available": True,
        "device": "cuda:0",
        "gpu_name": torch.cuda.get_device_name(0),
        "cuda_runtime": torch.version.cuda,
        "torch_version": torch.__version__,
        "vram_gb": round(props.total_memory / 1024**3, 3),
        "capability": list(torch.cuda.get_device_capability(0)),
        "fail_fast_if_cpu": True,
        "silent_cpu_fallback": False,
    }
    if "2060" not in info["gpu_name"] and props.total_memory < 5.5 * 1024**3:
        raise RuntimeError(f"UNEXPECTED_GPU:{info['gpu_name']}")
    return info


def bootstrap_kltn_namespace() -> None:
    """Expose kltn `src.hybrid` without shadowing student `src.prediction`."""
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    import src

    kltn_src = str(KLTN / "src")
    if kltn_src not in list(src.__path__):
        src.__path__.append(kltn_src)


def split_paths() -> dict[str, Path]:
    return {
        "uci_inner": KLTN / "artifacts" / "hybrid" / "phase1" / "splits" / "uci_inner.parquet",
        "uci_outer": KLTN / "artifacts" / "hybrid" / "phase1" / "splits" / "uci_outer.parquet",
        "oulad_inner": KLTN / "artifacts" / "hybrid" / "phase1" / "splits" / "oulad_inner.parquet",
        "oulad_outer": KLTN / "artifacts" / "hybrid" / "phase1" / "splits" / "oulad_outer.parquet",
    }


def verify_split_hashes() -> dict[str, str]:
    paths = split_paths()
    observed = {key: sha256_file(paths[key]) for key in ("uci_inner", "oulad_inner")}
    for key, expected in SPLIT_HASHES_EXPECTED.items():
        if observed[key] != expected:
            raise RuntimeError(f"SPLIT_HASH_MISMATCH:{key}:{observed[key]}:{expected}")
    return observed


def outer_test_ids(domain: str) -> set[str]:
    """Load outer-fold-0 test IDs only to exclude them. Never used as labels."""
    import pandas as pd

    path = split_paths()[f"{domain}_outer"]
    frame = pd.read_parquet(path)
    col = "record_id"
    return set(frame.loc[frame.outer_fold == DEVELOPMENT_OUTER_FOLD, col].astype(str))


def assert_no_outer(ids: Iterable[str], domain: str) -> None:
    blocked = set(map(str, ids)) & outer_test_ids(domain)
    if blocked:
        raise RuntimeError(f"OUTER_FIREWALL_VIOLATION:{domain}:{len(blocked)}")


def assert_disjoint(*groups: Iterable[str]) -> None:
    sets = [set(map(str, group)) for group in groups]
    for i, left in enumerate(sets):
        for right in sets[i + 1 :]:
            overlap = left & right
            if overlap:
                raise RuntimeError(f"PARTITION_OVERLAP:{len(overlap)}")


def seed_everything(seed: int) -> None:
    import random

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def run_metadata(**kwargs: Any) -> dict[str, Any]:
    payload = {
        "timestamp": utc_now(),
        "git_commit": git_commit(),
        "branch": git_branch(),
        "outer_test_used": False,
        "development_outer_fold": DEVELOPMENT_OUTER_FOLD,
        "authority_untouched": True,
    }
    payload.update(kwargs)
    return payload
