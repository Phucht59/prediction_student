"""Phase 3 shared constants, hashes, and C0 topology lock."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from .protocol import ART as PHASE2_ART
from .protocol import ROOT, sha256_file

PHASE3 = ROOT / "artifacts" / "hybrid_vnext" / "phase3"
REPORTS3 = ROOT / "reports" / "hybrid_vnext" / "phase3"
RUNS3 = PHASE3 / "runs"
EXPECTED_P2 = {
    "SELECTED_TOPOLOGY.json": "1fe018e702caff917b5558b054f8ec45fd90d02d8de6eb20476a02b49c7041f5",
    "PROTOCOL_LOCK.json": "a0a62a4bc8a387740d856fe0d74a5520ae7613b15dd4897886616618e570fd83",
}
PHASE2_UCI_REF = 0.725022
PHASE2_UCI_STD = 0.026282
PHASE2_OULAD_REF = 0.825498
SEEDS = (42, 1201, 2026)
INNER_FOLDS = (0, 1, 2)
SCREEN_FOLD = 0
SCREEN_SEED = 42
UCI_OUTER_FOLDS = (0, 1, 2, 3, 4)
OULAD_OUTER_FOLDS = (0, 1, 2)
# 1-fold UCI is noisier than robust mean; document before ranking.
UCI_1FOLD_FLOOR = 0.685129 - 0.015
UCI_ROBUST_FLOOR = PHASE2_UCI_REF - 0.005
UCI_STD_CEILING = PHASE2_UCI_STD + 0.010
TOPOLOGY_SPEC = {
    "architecture_id": "C0",
    "public_model_class": "Hybrid",
    "temporal_path": "parallel",
    "fusion": "softmax_3way",
    "availability": "[1, temporal_available, temporal_available]",
    "cnn_blocks": 2,
    "cnn_kernel_size": 2,
    "cnn_dilations": [1, 2],
    "bilstm_layers": 1,
    "pooling": "masked_mean_max",
    "head": "binary_logit",
}


def topology_hash() -> str:
    return hashlib.sha256(json.dumps(TOPOLOGY_SPEC, sort_keys=True).encode()).hexdigest()


def verify_phase2_locks() -> dict[str, str]:
    observed = {}
    for name, expected in EXPECTED_P2.items():
        path = PHASE2_ART / name
        digest = sha256_file(path)
        observed[name] = digest
        if digest != expected:
            raise RuntimeError(f"PHASE2_HASH_MISMATCH:{name}:{digest}:{expected}")
    selected = json.loads((PHASE2_ART / "SELECTED_TOPOLOGY.json").read_text(encoding="utf-8"))
    if selected["architecture_id"] != "C0" or selected["temporal_path"] != "parallel":
        raise RuntimeError("PHASE2_TOPOLOGY_REGRESSION")
    if selected["fusion_contract"]["type"] != "softmax_3way":
        raise RuntimeError("PHASE2_FUSION_REGRESSION")
    return observed
