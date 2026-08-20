"""Machine-local roots. No hardcoded C:\\hufit\\kltn."""
from __future__ import annotations

import os
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = PACKAGE_ROOT.parents[1]


def _env_path(name: str, default: Path) -> Path:
    raw = os.environ.get(name)
    return Path(raw).expanduser().resolve() if raw else default.resolve()


DATA_ROOT = _env_path("DATA_ROOT", PROJECT_ROOT / "data" / "raw")
ARTIFACT_ROOT = _env_path("ARTIFACT_ROOT", PROJECT_ROOT / "artifacts" / "research" / "hybrid_superiority_v2")
REPORT_ROOT = PROJECT_ROOT / "reports" / "research" / "hybrid_superiority_v2"
PROTOCOL_ROOT = PROJECT_ROOT / "protocols" / "hybrid_superiority_v2"
CONFIG_ROOT = PROJECT_ROOT / "configs" / "research" / "hybrid_superiority_v2"

CACHE_DIR = ARTIFACT_ROOT / "cache"
MANIFEST_DIR = ARTIFACT_ROOT / "manifests"
CHECKPOINT_DIR = ARTIFACT_ROOT / "checkpoints"
OOF_DIR = ARTIFACT_ROOT / "oof"
METRIC_DIR = ARTIFACT_ROOT / "metrics"
STAT_DIR = ARTIFACT_ROOT / "stats"
FIGURE_DIR = ARTIFACT_ROOT / "figures"
RUN_DIR = ARTIFACT_ROOT / "runs"


def ensure_dirs() -> None:
    for path in (
        CACHE_DIR,
        MANIFEST_DIR,
        CHECKPOINT_DIR,
        OOF_DIR,
        METRIC_DIR,
        STAT_DIR,
        FIGURE_DIR,
        RUN_DIR,
        REPORT_ROOT,
        PROTOCOL_ROOT,
        CONFIG_ROOT,
    ):
        path.mkdir(parents=True, exist_ok=True)
