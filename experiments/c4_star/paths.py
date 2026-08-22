"""C4-STAR artifact roots. Reuses locked hybrid_superiority_v2 data cache."""
from __future__ import annotations

import os
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = PACKAGE_ROOT.parents[1]


def _env_path(name: str, default: Path) -> Path:
    raw = os.environ.get(name)
    return Path(raw).expanduser().resolve() if raw else default.resolve()


DATA_ROOT = _env_path("DATA_ROOT", PROJECT_ROOT / "data" / "raw")
PARENT_ARTIFACT = PROJECT_ROOT / "artifacts" / "research" / "hybrid_superiority_v2"
ARTIFACT_ROOT = _env_path("C4_ARTIFACT_ROOT", PROJECT_ROOT / "artifacts" / "research" / "c4_star_v2_1")
REPORT_ROOT = PROJECT_ROOT / "reports" / "c4_star_v2_1"
PROTOCOL_ROOT = PROJECT_ROOT / "protocols" / "c4_star_v2_1"
CONFIG_ROOT = PROJECT_ROOT / "configs" / "research" / "c4_star_v2_1"

CACHE_DIR = PARENT_ARTIFACT / "cache"
MANIFEST_DIR = ARTIFACT_ROOT / "manifests"
CHECKPOINT_DIR = ARTIFACT_ROOT / "checkpoints"
OOF_DIR = ARTIFACT_ROOT / "oof"
METRIC_DIR = ARTIFACT_ROOT / "metrics"
STAT_DIR = ARTIFACT_ROOT / "stats"
FIGURE_DIR = ARTIFACT_ROOT / "figures"
RUN_DIR = ARTIFACT_ROOT / "runs"
LOG_DIR = ARTIFACT_ROOT / "logs"
STATE_PATH = ARTIFACT_ROOT / "state.json"
HEALTH_PATH = ARTIFACT_ROOT / "gpu_health.json"


def ensure_dirs() -> None:
    for path in (
        ARTIFACT_ROOT,
        MANIFEST_DIR,
        CHECKPOINT_DIR,
        OOF_DIR,
        METRIC_DIR,
        STAT_DIR,
        FIGURE_DIR,
        RUN_DIR,
        LOG_DIR,
        REPORT_ROOT,
        PROTOCOL_ROOT,
        CONFIG_ROOT,
    ):
        path.mkdir(parents=True, exist_ok=True)
