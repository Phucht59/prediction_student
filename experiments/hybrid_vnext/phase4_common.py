"""Phase 4 constants: one C0, five OULAD states, active baselines without XGB."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from .phase3_common import TOPOLOGY_SPEC, topology_hash, verify_phase2_locks
from .protocol import ROOT, sha256_file

PHASE4 = ROOT / "artifacts" / "hybrid_vnext" / "phase4"
REPORTS4 = ROOT / "reports" / "hybrid_vnext" / "phase4"
RUNS4 = PHASE4 / "runs"
SEEDS = (42, 1201, 2026)
INNER_FOLDS = (0, 1, 2)
SCREEN_FOLD = 0
SCREEN_SEED = 42
UCI_STATES = ("S0", "S1", "S2")
OULAD_STATES = ("20pct", "35pct", "50pct", "75pct", "100pct")
OULAD_EARLY = ("20pct", "35pct", "50pct", "75pct")
ACTIVE_FAMILIES = ("LR", "DT", "RF", "SVM", "MLP")
SHARED_STRUCTURAL = {"d_fuse": 128, "cnn_channels": 64, "bilstm_hidden": 128}
PHASE3_HPO = {
    "oulad": {
        "lr": 0.00011844319751820385,
        "weight_decay": 0.0007114476009343421,
        "dropout": 0.31959818254342154,
        "batch_size": 128,
        "pos_weight_multiplier": 0.7790418060840998,
        "entropy_floor_coefficient": 0.005,
    },
    "uci": {
        "lr": 8.605034792033103e-05,
        "weight_decay": 0.0032859708169642424,
        "dropout": 0.4061978796339918,
        "batch_size": 32,
        "pos_weight_multiplier": 1.1830880728874675,
        "entropy_floor_coefficient": 0.002,
    },
}
STAGE_ORDER = {"uci": list(UCI_STATES), "oulad": list(OULAD_STATES)}


def digest_obj(payload) -> str:
    return hashlib.sha256(json.dumps(payload, sort_keys=True, default=str).encode()).hexdigest()
